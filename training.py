# add root project dir to import path
from data.datasets.allclear.dataset import AllClearDataset
from data.datasets.sen12mscrts.data.dataLoader_MOD import FastSEN12
from torch.utils.data import DataLoader
from flow_matching.trainers import flow_matching_guided_train_step
import torch
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))


# read configuration file

# setup dataloader from config
TrainData = DataLoader
ValidationData = DataLoader

# setup model

# setup optimizer

# setup training from config (batch size, epochs, ...)

# load latest weights from ckpt and load best metrics from best checkpoint
initialEpoch = 0
maxEpochs = 100
validationFreq = 1

#  launch training
for epoch in range(initialEpoch, maxEpochs):
    # set model to training mode
    # model.train()

    for batch in TrainData:
        flow_matching_guided_train_step(batch)
        pass

    if epoch % validationFreq == 0:
        # test loop on val set
        for batch in ValidationData:
            # pass to flow matching eval step
            pass
