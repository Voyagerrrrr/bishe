import argparse
import torch
import os
import sys
sys.path.append('..')
import training.trainer as trainer

from datasets.dataset_utils import make_dataloaders
from torchpack.utils.config import configs 

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--uncertainty_method', type=str, required=False, default='none', help='Uncertainty estimation method to be used. default=none. Options: MC Dropout')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    print(f'\n{configs}\n')
    print('Training config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))
    print('Visualize: {}'.format(args.visualize))

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    dataloaders = make_dataloaders(debug=args.debug)

    if args.uncertainty_method in ['dropout']:
        print('\n------------\nDropout training\n------------\n')
        trainer.do_train(dataloaders, debug=args.debug, visualize=args.visualize)
    else:
        print("\n-----------------------\nNo uncertainty training\n-----------------------\n")
        trainer.do_train(dataloaders, debug=args.debug, visualize=args.visualize)
