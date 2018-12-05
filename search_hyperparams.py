"""Peform hyperparemeters search"""
import argparse
import os
from subprocess import check_call
import sys
import numpy as np

from src.cnn_ws.utils import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--restore_file', default=None,
                    help='Directory of model to load')
parser.add_argument('--gpu_id', default='0', help="index of GPU to use")

def launch_training_job(parent_dir, restore_file, job_name, params, gpu_id):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    if restore_file is not None: 
        cmd = "{python} -W ignore train.py --dont_save_model --model_dir={model_dir} --restore_file {restore_file} --gpu_id {gpu_id}".format(python=PYTHON, model_dir=model_dir, restore_file=restore_file, gpu_id=gpu_id)
    else: 
        # Launch training with this config
        cmd = "{python} -W ignore train.py --dont_save_model --model_dir={model_dir} --gpu_id {gpu_id}".format(python=PYTHON, model_dir=model_dir, gpu_id=gpu_id)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = ["600000:1e-3,1000000:1e-4", 
                      "600000:1e-4,1000000:1e-5",
                      "600000:1e-5,1000000:1e-6",
                      "600000:1e-6,1000000:1e-7",
                      "600000:1e-7,1000000:1e-8"]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate
        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        print('launching training job...')
        launch_training_job(args.parent_dir, args.restore_file, job_name, params, args.gpu_id)
