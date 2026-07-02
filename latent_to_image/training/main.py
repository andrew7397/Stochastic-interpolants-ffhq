import argparse
import os
import sys

import satflow.common.config_helpers as helpers
from satflow.common.utils import is_deterministic_learning

from .multi_model import MultiModelTraining
from .single_model import SingleModelTraining

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

arg_parser = argparse.ArgumentParser(description='Train TODO.')
arg_parser.add_argument('json_config_dir', metavar='<JSON config directory>',
                        help='Path to the JSON directory containing the configurations to run this training script.')
args = arg_parser.parse_args()

if __name__ == '__main__':
    
    # Load config
    config = helpers.load_config_from_JSON(args.json_config_dir)

    if config.model.single_model or is_deterministic_learning(config.interpolant.type_of_learning):
        trainer = SingleModelTraining(config)
    else:
        trainer = MultiModelTraining(config)

    trainer.run()
