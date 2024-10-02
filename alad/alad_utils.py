import os
import logging
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_results_dir(dataset_name: str, allow_zz: bool, random_seed: int):
    """
    Create dir for saving results. 
    Returns dir path.
    """
    dir_path = f"results/{dataset_name}/alad_dzz{allow_zz}/seed_{random_seed}"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def create_logger(dataset_name: str, allow_zz: bool, random_seed: int):
    """
    Create logger that will print both to log_file and stdout. 
    Returns logger.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f"results/{dataset_name}/alad_dzz{allow_zz}/seed_{random_seed}/log.log",
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


def save_plot_losses(loss_dict, save_path):
    """
    Save plot of ALAD train losses
    """
    plt.figure(figsize=(10, 6))

    plt.plot(loss_dict["epoch"], loss_dict['gen'], label='Gen')
    plt.plot(loss_dict["epoch"], loss_dict['enc'], label='Enc')
    plt.plot(loss_dict["epoch"], loss_dict['dis'], label='Dis')
    plt.plot(loss_dict["epoch"], loss_dict['dis_xz'], label='DisXZ')
    plt.plot(loss_dict["epoch"], loss_dict['dis_xx'], label='DisXX')
    plt.plot(loss_dict["epoch"], loss_dict['dis_zz'], label='DisZZ')
    plt.plot(loss_dict["epoch"], loss_dict['val_loss'], label='Val loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('ALAD train losses')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    

def print_parameters(logger, **kwargs):
    logger.info(f"Parameters:")
    for key, value in kwargs.items():
        logger.info(f"{key} : {value}")
    logger.info(f"\n")

