import signal
from typing import Literal, Optional, Union

import torch
import wandb
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchvision.transforms import Compose

from satflow.common.config import InterpolantConfig
from satflow.common.interflow.prior import GMM, SimpleNormal
from satflow.common.interflow.stochastic_interpolant import (Interpolant,
                                                             PFlowIntegrator,
                                                             SDEIntegrator)
from satflow.common.logging import (log_loss, log_metrics_wandb, log_progress,
                                    log_wandb, log_wandb_images)
from satflow.common.model_entry import ModelEntry
from satflow.common.utils import multispectral_to_rgb


class TerminationHandler():
    """
    Simple termination handler to allow graceful termination upon receiving
    SIGINT (ctrl-c) and SIGTERM (slurm process termination). This operation
    is fundamental to avoid inconsistencies when saving model checkpoints and
    therefore to not lose training progress.
    """

    def __init__(self):
        self.requested_termination = False
        signal.signal(signal.SIGINT, self.sig_intercept)
        signal.signal(signal.SIGTERM, self.sig_intercept)

    def sig_intercept(self, sig_num, frame):
        self.requested_termination = True


def generate_samples(
    interpolant_config: InterpolantConfig,
    model_list: list[ModelEntry],
    interpolant: Interpolant,
    initial_distr: Union[SimpleNormal, GMM],
    n_gen_samples: int,
    n_sampling_steps: int,
    inverse_preprocess_transform: Compose,
    device: torch.device
):

    with torch.no_grad():

        type_of_learning = interpolant_config.type_of_learning
        eps = interpolant_config.eps
        lower_ts = interpolant_config.lower_ts
        upper_ts = interpolant_config.upper_ts
        sampler_method = interpolant_config.sampler_method
        atol = interpolant_config.atol
        rtol = interpolant_config.rtol

        stochastic = model_list[0].stochastic
        single_model = True if model_list[0].name == "single_model" else False

        x0s = initial_distr(n_gen_samples).to(
            device)                     # Initial samples

        # sample_only = False --> return the samples and their log-probabilities
        sample_only = True

        if stochastic:
            model_A = model_list[0].model
            model_B = None if single_model else model_list[1].model

            sampler = SDEIntegrator(
                model_A,
                model_B,
                single_model,
                type_of_learning,
                eps,
                interpolant,
                1,
                (lower_ts, upper_ts),
                n_sampling_steps,
                1
            )

            x1s_sde = sampler.rollout_forward(x0s, sampler_method)[-1]
            x1s_sde = inverse_preprocess_transform(x1s_sde).detach().cpu()
            x1s_sde_list = [x1s_sde[i] for i in range(x1s_sde.shape[0])]

            if not sample_only:
                x0s_sdeflow, _ = sampler.rollout_likelihood(x1s_sde.to(device))
            else:
                x0s_sdeflow = torch.zeros(x1s_sde.shape).to(device)

            return x1s_sde_list, x0s_sdeflow

        sampler = PFlowIntegrator(
            model_list[0].model,
            sampler_method,
            interpolant,
            (lower_ts, upper_ts),
            n_sampling_steps,
            atol,
            rtol,
            sample_only,
            False
        )

        # Detect where the distribution is to avoid device mismatch
        dist_device = next(initial_distr.buffers()).device
        logp0 = initial_distr.log_prob(x0s.to(dist_device)).to(device)
        xfs_ode, dlogp_ode = sampler.rollout(x0s)
        logpx_ode_x1s = logp0 + \
            (dlogp_ode[-1].squeeze() if dlogp_ode is not None else 0)

        x1s_ode = xfs_ode[-1]
        x1s_ode = inverse_preprocess_transform(x1s_ode).detach().cpu()
        x1s_ode_list = [x1s_ode[i] for i in range(x1s_ode.shape[0])]

        return x1s_ode_list, logpx_ode_x1s


def batch_train_step(
    batch,
    model_list: list[ModelEntry],
    interpolant: Interpolant,
    initial_distr: Union[SimpleNormal, GMM],
    clip_grad_norm: bool,
    preprocess_transform: Compose,
    RGB_case: bool,
    device: torch.device
):
    # Reorder channels only for multispectral satellite data, not for RGB
    if not RGB_case:
        batch = batch[:, [3, 2, 1]].contiguous()

    x1s = preprocess_transform(batch)

    bs = x1s.shape[0]

    x0s = initial_distr(bs).to(device)                     # Initial samples
    ts = torch.rand(size=(bs,)).to(device)

    tot_loss = torch.tensor(0.0).to(device)
    dict_losses: dict[str, float] = {}
    dict_norms: dict[str, list[float]] = {}
    dict_lrs: dict[str, float] = {}

    for model_entry in model_list:
        name = model_entry.name
        model = model_entry.model
        opt = model_entry.opt
        sch = model_entry.scheduler
        ema = model_entry.ema
        stochastic = model_entry.stochastic
        opt.zero_grad()
        loss = model_entry.loss(model, x0s, x1s, ts, interpolant)

        # Case stochastic with single model
        if name == "single_model" and stochastic:
            loss_A, loss_B = loss
            (loss_A + loss_B).backward()

            dict_losses.update({
                f"{model_entry.name}_loss_A": loss_A.item(),
                f"{model_entry.name}_loss_B": loss_B.item()
            })

            tot_loss += (loss_A + loss_B) / 2.0
        else:
            loss.backward()

            dict_losses[model_entry.name] = loss.item()

            tot_loss += loss

        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True)

        total_norm = 0.0
        n_params = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)        # L2 norm
                total_norm += param_norm.item() ** 2
                n_params += 1

        mean_norm = total_norm / n_params

        dict_norms[model_entry.name] = [total_norm, mean_norm]

        current_lr = opt.param_groups[0]['lr']
        dict_lrs[model_entry.name] = current_lr

        opt.step()

        if sch is not None:
            sch.step()

        if ema is not None:
            ema.update_parameters(model)

    tot_loss /= float(len(model_list))

    return tot_loss.item(), dict_losses, dict_norms, dict_lrs


def batch_eval_step(
    batch,
    model_list: list[ModelEntry],
    interpolant: Interpolant,
    initial_distr: Union[SimpleNormal, GMM],
    preprocess_transform: Compose,
    RGB_case: bool,
    device: torch.device
):

    with torch.no_grad():

        if not RGB_case:
            batch = batch[:, [3, 2, 1]]

        x1s = preprocess_transform(batch)

        bs = x1s.shape[0]

        # Initial samples
        x0s = initial_distr(bs).to(device)
        ts = torch.rand(size=(bs,)).to(device)

        tot_loss = torch.tensor(0.0).to(device)
        dict_losses: dict[str, float] = {}

        for model_entry in model_list:
            name = model_entry.name
            model = model_entry.model
            opt = model_entry.opt
            stochastic = model_entry.stochastic
            opt.zero_grad()
            loss = model_entry.loss(model, x0s, x1s, ts, interpolant)

            if name == "single_model" and stochastic:
                loss_A, loss_B = loss

                dict_losses.update({
                    f"{model_entry.name}_loss_A": loss_A.item(),
                    f"{model_entry.name}_loss_B": loss_B.item()
                })

                tot_loss += (loss_A + loss_B) / 2.0
            else:
                dict_losses[model_entry.name + '_loss_A'] = loss.item()

                tot_loss += loss

        tot_loss /= float(len(model_list))

    return tot_loss.item(), dict_losses


def fit(
    model_list: list[ModelEntry],
    interpolant: Interpolant,
    interpolant_config: InterpolantConfig,
    initial_distr: Union[SimpleNormal, GMM],
    training_data: DataLoader,
    validation_data: DataLoader,
    val_subset_metrics: DataLoader,
    clip_grad_norm: bool,
    preprocess_transform: Compose,
    inverse_preprocess_transform: Compose,
    max_epochs: int,
    metrics: dict[str, Union[Metric, MetricCollection]],
    visual_freq_steps: int,
    validation_freq_steps: int,
    wandb_log_run: wandb.Run,
    device: torch.device,
    visual_samples: dict[Literal['train', 'val'], list[torch.Tensor]],
    initial_epoch: int = 0,
    initial_step: int = 0,
    initial_best: Optional[float] = None,
    RGB_case: bool = False,
    verbose: bool = True,
):
    '''
    Runs a training loop on the latent decoder model and evaluates the model
    on the given validation set every given batches.
    '''

    termination_handler = TerminationHandler()

    train_batch_size = training_data.batch_size
    validation_batch_size = validation_data.batch_size

    num_training_batches = len(training_data)
    num_validation_batches = len(validation_data)

    num_training_samples = len(training_data.dataset)
    num_validation_samples = len(validation_data.dataset)

    # Validation metrics
    validation_metrics = {}
    if 'fid' in metrics:
        validation_metrics['fid'] = metrics['fid'].clone()
    if 'collection' in metrics:
        validation_metrics['collection'] = metrics['collection'].clone()

    # global_step used as x-axis when logging.
    # Computed as the number of processed samples
    # Initialised from latest recorded global_step in case of chekpoint
    global_step = initial_step
    best_loss_fn = initial_best

    if global_step == 0:
        log_wandb_images(
            wandb_log_run, visual_samples['train'], "train", RGB_case, 0)
        log_wandb_images(wandb_log_run, visual_samples['val'], "val", RGB_case, 0)

    for epoch in range(initial_epoch, max_epochs):
        print(f'\n### epoch {epoch} / {max_epochs} ###')

        for model_entry in model_list:
            model_entry.model.train()

        running_train_loss_fn = 0.0

        for train_batch_index, train_batch in enumerate(training_data):
            # Handle datasets that return dicts (e.g., FFHQ)
            if isinstance(train_batch, dict):
                train_batch = train_batch['input_images']

            tot_loss_train, dict_losses_train, dict_norms_train, dict_lrs_train = batch_train_step(
                train_batch.to(device),
                model_list,
                interpolant,
                initial_distr,
                clip_grad_norm,
                preprocess_transform,
                RGB_case,
                device
            )

            running_train_loss_fn += tot_loss_train

            # step computed as number of processed samples
            # step relative to current epoch (assumes constant batch size)
            assert train_batch_size is not None, "Training dataloader must have a defined batch size"
            current_train_step = (train_batch_index + 1) * train_batch_size

            global_step += 1

            # Save latest checkpoint periodically instead of every step to avoid slowing down training
            # (especially when saving to Google Drive)
            if global_step % 100 == 0:
                for model_entry in model_list:
                    dict_ckpt = {
                        'epoch': epoch,
                        'step': global_step,
                        'model': model_entry.model.state_dict(),
                        'optimizer': model_entry.opt.state_dict()
                    }

                    if model_entry.scheduler is not None:
                        dict_ckpt["scheduler"] = model_entry.scheduler.state_dict()

                    if model_entry.ema is not None:
                        dict_ckpt["ema"] = model_entry.ema.state_dict()

                    torch.save(dict_ckpt, model_entry.ckpt_latest)

            if verbose:
                log_progress('Training', 'train', tot_loss_train,
                             dict_losses_train, global_step, dict_norms_train, dict_lrs_train, (max_epochs * num_training_batches))

            if wandb_log_run is not None:
                log_wandb(wandb_log_run, 'train', tot_loss_train,
                          dict_losses_train, global_step, dict_norms_train, dict_lrs_train)

            if termination_handler.requested_termination:
                print('\nRequested termination. Exiting!')
                exit()

            # --------- Generate visual samples ---------
            if global_step % visual_freq_steps == 0:

                print("\n\n------- Generating samples -------\n\n")

                predicted_list_visual_samples, _ = generate_samples(
                    interpolant_config, model_list, interpolant,
                    initial_distr, len(visual_samples['train']), 
                    interpolant_config.n_visual_sampling_steps,
                    inverse_preprocess_transform, device
                )

                log_wandb_images(
                    wandb_log_run, predicted_list_visual_samples, "generated", RGB_case, global_step)

            if global_step % validation_freq_steps == 0:

                print("\n\n------- Validation phase -------\n\n")

                running_validation_loss_fn = 0.0

                if 'fid' in validation_metrics:
                    validation_metrics['fid'].reset()
                if 'collection' in validation_metrics:
                    validation_metrics['collection'].reset()

                dict_tot_losses = {}
                for model_entry in model_list:
                    model_entry.model.eval()
                    dict_tot_losses[model_entry.name] = 0.0

                if verbose:
                    print('')

                for val_batch_index, batch_val_data in enumerate(validation_data):
                    # Handle datasets that return dicts (e.g., FFHQ)
                    if isinstance(batch_val_data, dict):
                        batch_val_data = batch_val_data['input_images']

                    tot_loss_val, dict_losses_val = batch_eval_step(
                        batch_val_data.to(device),
                        model_list,
                        interpolant,
                        initial_distr,
                        preprocess_transform,
                        RGB_case,
                        device
                    )

                    running_validation_loss_fn += tot_loss_val

                    assert validation_batch_size is not None, "Validation dataloader must have a defined batch size"
                    current_validation_step = (
                        val_batch_index + 1) * validation_batch_size

                    if verbose:
                        log_progress('Validating', 'val', tot_loss_val, dict_losses_val,
                                     current_validation_step, total_steps=num_validation_samples)

                    for model_entry in model_list:
                        for key in dict_losses_val.keys():
                            if model_entry.name in key:
                                dict_tot_losses[model_entry.name
                                                ] += dict_losses_val[key]

                    if termination_handler.requested_termination:
                        print('Requested termination. Exiting!')
                        exit()

                validation_loss_fn = running_validation_loss_fn / num_validation_batches
                for model_entry in model_list:
                    dict_tot_losses[model_entry.name
                                    ] /= num_validation_batches

                if wandb_log_run is not None:
                    log_wandb(wandb_log_run, 'val', validation_loss_fn,
                              dict_tot_losses, step=global_step)

                if best_loss_fn is None or validation_loss_fn < best_loss_fn:
                    best_loss_fn = validation_loss_fn

                    for model_entry in model_list:

                        dict_ckpt = {
                            "epoch": epoch,
                            "step": global_step,
                            "best_loss": best_loss_fn,
                            "model": model_entry.model.state_dict(),
                            "optimizer": model_entry.opt.state_dict(),
                        }

                        if model_entry.scheduler is not None:
                            dict_ckpt["scheduler"] = model_entry.scheduler.state_dict()

                        if model_entry.ema is not None:
                            dict_ckpt["ema"] = model_entry.ema.state_dict()

                        torch.save(dict_ckpt, model_entry.ckpt_best)

                    # --------------- Metrics computation ---------------
                    if 'fid' in validation_metrics:
                        validation_metrics['fid'].reset()
                    if 'collection' in validation_metrics:
                        validation_metrics['collection'].reset()

                    for batch_val_data in val_subset_metrics:
                        # Handle datasets that return dicts (e.g., FFHQ)
                        if isinstance(batch_val_data, dict):
                            batch_val_data = batch_val_data['input_images']

                        predicted_list_metrics_samples, predicted_likelihood_samples = generate_samples(
                            interpolant_config, model_list, interpolant,
                            initial_distr, batch_val_data.shape[0], 
                            interpolant_config.n_metric_sampling_steps,
                            inverse_preprocess_transform, device
                        )

                        batch_fake_val_data = torch.stack(
                            predicted_list_metrics_samples)

                        batch_val_data_zero_one, batch_val_data_minus_one_one = multispectral_to_rgb(
                            batch_val_data, device, RGB_case=RGB_case)
                        batch_fake_val_data_zero_one, batch_fake_val_data_minus_one_one = multispectral_to_rgb(
                            batch_fake_val_data, device, RGB_case=RGB_case)

                        if 'fid' in validation_metrics:
                            validation_metrics['fid'].update(
                                batch_val_data_zero_one, real=True)
                            validation_metrics['fid'].update(
                                batch_fake_val_data_zero_one, real=False)

                        if 'collection' in validation_metrics:
                            validation_metrics['collection'].update(
                                batch_fake_val_data_minus_one_one, batch_val_data_minus_one_one)

                    validation_metrics_results = {}
                    if 'fid' in validation_metrics:
                        validation_metrics_results['fid'] = validation_metrics['fid'].compute()
                    if 'collection' in validation_metrics:
                        validation_metrics_results.update(validation_metrics['collection'].compute())

                    if validation_metrics_results:
                        print(validation_metrics_results)
                        log_metrics_wandb(
                            wandb_log_run, validation_metrics_results, global_step)

                    if 'fid' in validation_metrics:
                        validation_metrics['fid'].reset()
                    if 'collection' in validation_metrics:
                        validation_metrics['collection'].reset()

                # set model back to training mode
                for model_entry in model_list:
                    model_entry.model.train()

        train_loss_fn = running_train_loss_fn / num_training_batches

        if verbose:
            print('')
            log_loss('Train final', 'train', train_loss_fn)
