'''
Created on Nov 4, 2018

@author: taivanbat

Script to run queries
'''
import os
import argparse

import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from skimage import io as img_io
import numpy as np
import json
from tqdm import tqdm

from src.cnn_ws.evaluation.retrieval import run_query
from src.cnn_ws.utils.save_load import my_torch_load, my_torch_save
from src.cnn_ws.models.myphocnet import PHOCNet

parser = argparse.ArgumentParser()
# eval arguments
parser.add_argument('--gpu_id', '-gpu', action='store',
                    type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                    default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--model_dir', default='models/base_model', help="Directory containing params.json")
parser.add_argument('--debug', default=False, help='Set to true if we want to debug')
parser.add_argument('--save_im_results', default=True, help='Set to true if we want to save top num_save_im results of search')
parser.add_argument('--num_save_im', default=10, help='how many images to save')

# TODO add ability to load wiener from words from json
parser.add_argument('--wiener_from_json', default=False)
args = parser.parse_args()
args.num_save_im = int(args.num_save_im) 
# if GPU is not available, use CPU
if not torch.cuda.is_available():
    args.gpu_id = None

def eval(args, params):
    # if candidates aren't known, make them
    if 'candidates.npy' not in os.listdir('.') or 'candidates_labels.json' not in os.listdir('.'):
        candidates, candidates_labels = make_candidates(args=args, params=params)
    else:
        # load candidates
        candidates = np.load('candidates.npy')

        # load candidates_labels
        with open('candidates_labels.json', 'r') as filehandle:
            candidates_labels = json.load(filehandle)

    # can make this such that it loads from a json file
    queries = ['und', 'ich', 'bin', 'eigen', 'besonders']

    results = run_query(args.num_save_im, candidates, candidates_labels, queries, args, params)

    # save the results. This is a num_queries x num_candidates sized matrix where the indices
    # of the most likely result is stored in column 1, and the second most likely is stored
    # in column 2 and so on
    np.save('results.npy', results)

def make_candidates(args=args, params=params):
    # load model, create candidates
    # TODO path to candidates should be an argument 
    # TODO specify whether we're loading wiener from json files
    path_to_cands = 'data/wiener/candidates/word_images'
    folders = os.listdir(path_to_cands)
    folders = [folder for folder in folders if folder != '.DS_Store']

    if args.debug:
        folders = folders[:2]

    # load model
    # 675 is what we have by default
    cnn = PHOCNet(n_out=675,
                  input_channels=1,
                  gpp_type='gpp',
                  pooling_levels=([1], [5]))

    cnn.init_weights()

    # TODO: make loading from model_dir 
    my_torch_load(cnn, 'PHOCNet_50k.pt')

    if args.gpu_id is not None:
        cnn.cuda(args.gpu_id[0])

    # set to eval mode
    cnn.eval()

    candidates = []
    candidates_labels = []

    with tqdm(total=len(folders)) as t:
        for i, folder in enumerate(folders):
            word_img_names = sorted(
                [elem for elem in os.listdir(os.path.join(path_to_cands, folder)) if elem.endswith('.jpg')])

            candidates_labels += ['_'.join([folder, word_img_name[:-4]]) for word_img_name in word_img_names]
            word_imgs = [img_io.imread(os.path.join(path_to_cands, folder, word_img_name))
                         for word_img_name in word_img_names]

            word_imgs = [torch.autograd.Variable(torch.from_numpy(word_img)).float() for word_img in word_imgs]

            if args.gpu_id is not None:
                word_imgs = [word_img.cuda(args.gpu_id[0]) for word_img in word_imgs]

            word_imgs = [word_img.unsqueeze(0) for word_img in word_imgs]
            word_imgs = [word_img.unsqueeze(0) for word_img in word_imgs]

            # get phoc representation of words
            word_phocs = [torch.sigmoid(cnn(word_img)) for word_img in word_imgs]
            word_phocs = [word_phoc.data.cpu().numpy().flatten() for word_phoc in word_phocs]
            word_phocs = np.vstack(word_phocs)
            # concatenate along the row dimension as a numpy array
            # if first in list of folders, candidates are just phocs from this folder
            if i == 0:
                candidates = np.vstack(word_phocs)
            else:
                candidates = np.vstack((candidates, word_phocs))

            t.update(1)

    # save candidates as npy file
    np.save('candidates.npy', candidates)

    # save candidates_labels as json file
    with open('candidates_labels.json', 'w') as filehandle:
        json.dump(candidates_labels, filehandle)

    return candidates, candidates_labels

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    eval(args=args, params=params)
