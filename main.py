import argparse
import os

from alad.alad import run



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ALAD', description='Run ALAD', epilog='Text at the bottom of help')

    # General
    parser.add_argument('dataset', choices=list(os.walk("data"))[0][1], help='the name of the dataset you want to run the experiments on')
    parser.add_argument('--epochs', nargs="?", type=int, default=0, help='number of epochs you want to train the dataset on')
    parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='batch size')
    parser.add_argument('--seed', nargs="?", type=int, default=42, help='random_seed')
    parser.add_argument('--enable_sm', action='store_true', help='enable TF summaries')
    parser.add_argument('--early stopping', nargs="?", type=int, default=42, help='early stopping, number of epochs without improvement')

    # ALAD
    parser.add_argument('--degree', nargs="?", type=int, default=1, help='degree for the L norm')
    parser.add_argument('--enable_dzz', action='store_true', help='enable dzz discriminator')


    run(parser.parse_args())
