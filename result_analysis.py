# The function to evaluate the performance of the trained model
from helper_functions import load_flags
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def put_trained_models_into_folders(mother_dir='models/'):
    """
    The function that put the trained models into folders as the name suggest
    :param: mother dir
    """
    for folder in os.listdir(mother_dir):
        cur_folder = os.path.join(mother_dir, folder)
        # Make sure this is a folder
        if not os.path.isdir(cur_folder) or 'complexity' not in folder:
            continue
        # Create a bunch of sub folders if not exist
        hyper_param_str = folder.split('complexity_')[-1]
        hyper_param_folder = os.path.join(mother_dir, hyper_param_str)
        # If this hyper_param exist, move this folder into it
        if not os.path.isdir(hyper_param_folder):
            os.makedirs(hyper_param_folder)
        # Move that folder down
        os.rename(cur_folder, os.path.join(hyper_param_folder, folder))
        
        # Example name: 'bruce_high_freq_return_ind_19583_complexity_500x7_lr_0.001_decay_0.2_reg_0.01_bs_128'

def get_compy_ind_insample_outsample_loss(model_folder):
    """
    :param: The full path of the modle folder containing the flags.obj file
    """
    flag = load_flags(model_folder)
    return flag.comp_ind, flag.best_training_loss, flag.best_validation_loss

def get_all_ind_insample_outsample_loss(mother_folder, save_name='comp_in_out_MSE.csv'):
    """
    Calls sub funcitons to get all the in_sample out_sample MSE into a csv file
    """
    compy_list = []
    # Get the company list
    for folder in os.listdir(mother_folder):
        cur_folder = os.path.join(mother_folder, folder)
        # Make sure this is a folder
        if not os.path.isdir(cur_folder) or not os.path.isfile(os.path.join(cur_folder, 'flags.obj')):
            print('Either this is not a folder or it has no flags.obj file')
            print('Name = ', cur_folder)
            continue
        # Initialize the dictionary of the current compay
        cur_comp = {}
        comp_ind, insample, outsample = get_compy_ind_insample_outsample_loss(cur_folder)
        cur_comp['comp_ind'] = comp_ind
        cur_comp['insample'] = insample
        cur_comp['outsample'] = outsample
        compy_list.append(cur_comp)

    # Make the company list into a csv file
    df = pd.DataFrame(compy_list)
    df.to_csv(os.path.join(mother_folder, '..', os.path.basename(mother_folder) + save_name))

def get_all_hyper_analysis(mother_folder='models/'):
    for folder in os.listdir(mother_folder):
        cur_folder = os.path.join(mother_folder, folder)
        if not os.path.isdir(cur_folder) or 'complexity' in folder:
            continue
        get_all_ind_insample_outsample_loss(cur_folder)

if __name__ == '__main__':
    # First step, after running a bunch of models, throw the models into same hyper-param folder
    #put_trained_models_into_folders()
    # Get the .csv file
    get_all_hyper_analysis()