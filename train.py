"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import pandas as pd
import numpy as np
import sys
# Torch

# Own
import flag_reader
import data_reader
from class_wrapper import Network
from model_maker import Forward
from helper_functions import put_param_into_folder, write_flags_and_BVE


def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)

    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader)
    total_param = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print("Total learning parameter is: %d"%total_param)
    
    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk)
    # put_param_into_folder(ntwk.ckpt_dir)


def train_all_models(gpu):
    """
    The aggregate training
    """
    for hidden_layer_num in range(7):
    #for hidden_layer_num in [1, 3, 5, 7]:
        for neurons in [20, 50, 100, 200, 500]:
            flags = flag_reader.read_flag()
            flags.linear = [flags.dim_x] + [neurons for i in range(hidden_layer_num)] + [flags.dim_y]
            flags.model_name = flags.data_set + '_complexity_{}x{}_lr_{}_decay_{}_reg_{}_bs_{}'.format(flags.linear[1], len(flags.linear) - 2, flags.lr, flags.lr_decay_rate, flags.reg_scale, flags.batch_size)
            print(flags.model_name)
            training_from_flag(flags)

if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()
    # Call the train from flag function
    #training_from_flag(flags)
    train_all_models(-1)
    #for i in range(4):
    #    get_list_comp_ind(i)
