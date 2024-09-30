import os
import logging
import sys

import tensorflow as tf
import numpy as np


def create_results_dir(dataset: str, allow_zz: bool, random_seed: int):
    dir_path = f"results/{dataset}/alad_dzz{allow_zz}/seed_{random_seed}"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def create_logger(dataset, allow_zz, random_seed):
    """
    Get logger that will print both to log_file and stdout
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f"results/{dataset}/alad_dzz{allow_zz}/seed_{random_seed}/log.log",
                        filemode='a')

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logger = logging.getLogger('alad')

    return logger


def batch_fill(testx, batch_size):
    """ Quick and dirty hack for filling smaller batch

    :param testx:
    :param batch_size:
    :return:
    """
    nr_batches_test = int(testx.shape[0] / batch_size)
    ran_from = nr_batches_test * batch_size
    ran_to = (nr_batches_test + 1) * batch_size
    size = testx[ran_from:ran_to].shape[0]
    new_shape = [batch_size - size] + list(testx.shape[1:])
    fill = np.ones(new_shape).astype(np.float32)
    return np.concatenate([testx[ran_from:ran_to], fill], axis=0), size


# def plot_model(model):
#     """
#     Plot keras model
#     """
#     tf.keras.utils.plot_model(model)

def print_parameters(logger, **kwargs):
    logger.info(f"Parameters:")
    for key, value in kwargs.items():
        logger.info(f"{key} : {value}")
    logger.info(f"\n")

