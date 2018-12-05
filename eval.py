'''
Created on Nov 4, 2018

@author: taivanbat

Script to run queries
'''
import os
import argparse
import logging

import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from skimage import io as img_io
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

from src.cnn_ws.evaluation.retrieval import run_query
from src.cnn_ws.utils.save_load import my_torch_load, my_torch_save
from src.cnn_ws.models.myphocnet import PHOCNet
from src.cnn_ws.utils import utils

parser = argparse.ArgumentParser()
# eval arguments
parser.add_argument('--gpu_id', '-gpu', action='store',
                    type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                    default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--model_dir', default='models/base_model', help="Directory containing params.json")
parser.add_argument('--debug', action='store_true', help='Set to true if we want to debug')
parser.add_argument('--save_im_results', default=True, help='Set to true if we want to save top num_save_im results of search')
parser.add_argument('--num_save_im', default=10, help='how many images to save')

# set if we're loading words from the wiener files
parser.add_argument('--wiener_from_json', action='store_true')

def eval(params):
    # if candidates aren't known, make them
    if 'candidates.npy' not in os.listdir(params.model_dir) or 'candidates_labels.json' not in os.listdir(params.model_dir):
        candidates, candidates_labels = make_candidates(params)
    else:
        # load candidates
        candidates = np.load(os.path.join(params.model_dir,'candidates.npy'))

        # load candidates_labels
        with open(os.path.join(params.model_dir,'candidates_labels.json'), 'r') as filehandle:
            candidates_labels = json.load(filehandle)

    # TODO enable handling situations where character is not in the character set
    # queries = ['und', 'ich', 'bin', 'eigen', 'besonders']
    with open('word_search.txt') as f: 
        queries = f.read().split()

    queries = [query.lower() for query in queries]

    results = run_query(params.num_save_im, candidates, candidates_labels, queries, params)

    # save the results. This is a num_queries x num_candidates sized matrix where the indices
    # of the most likely result is stored in column 1, and the second most likely is stored
    # in column 2 and so on
    np.save(os.path.join(params.model_dir, 'results.npy'), results)

def make_candidates(params):
    # load model, create candidates
    # TODO path to candidates should be an argument 
    # TODO specify whether we're loading wiener from json files
    if params.wiener_from_json:
        # this is the base path
        # within it are wiener_images_cropped (containing original images with original size)
        # and also wiener_images_segmented (containing the .json files with bounding box information)
        path_to_cands = '/specific/disk1/home/adiel/wiener'
    else:
        path_to_cands = 'data/wiener/candidates/word_images'

    # load model
    # 675 is what we have by default
    cnn = PHOCNet(n_out=690,
                  input_channels=1,
                  gpp_type=params.gpp_type,
                  pooling_levels=params.pooling_levels)

    cnn.init_weights()

    my_torch_load(cnn, os.path.join(params.model_dir, 'PHOCNET_best.pt'))

    if params.gpu_id is not None:
        cnn.cuda(params.gpu_id[0])

    # set to eval mode
    cnn.eval()

    candidates = []
    candidates_labels = []

    # TODO!!!
    # enable loading from json files
    # want to store:
    # PHOC representation of each word
    # coordinates in original image of each word
    # as well as the name of the image to which it belongs

    if params.wiener_from_json:
        cropped_img_folder = os.path.join(path_to_cands, 'wiener_images_cropped')
        segmented_img_folder = os.path.join(path_to_cands, 'wiener_images_segmented')

        word_phocs = []
        candidates_labels = []

        # get list of images
        collections = [collection for collection in os.listdir(cropped_img_folder) if collection not in ['.', '..']]

        for coll_idx, collection in enumerate(collections):
            # get names of all images
            page_names = os.listdir(os.path.join(cropped_img_folder, collection))
            page_names = sorted([page_name for page_name in page_names if page_name.endswith('.jpg')])

            logging.info('going over collection %s with %s images', collection, str(len(page_names)))
            
            # go over up to a 100 images

            num_pages = min(len(page_names),100)

            with tqdm(total=num_pages) as t:
                # iterate over pages
                for page_idx, page_name in enumerate(page_names[:num_pages]):
                    # numpy --> page_img.shape is in row x col
                    page_img = img_io.imread(os.path.join(cropped_img_folder, collection, page_name))

                    ratio_yx = float(page_img.shape[0])/page_img.shape[1]

                    if ratio_yx < float(1200)/900:
                        r = page_img.shape[1]/float(900)
                    else:
                        r = page_img.shape[0]/float(1200)

                    with open(os.path.join(segmented_img_folder, collection, page_name[:-4] + '.json')) as json_f:
                        json_info = json.load(json_f)

                    word_imgs = []
                    bboxes = []
                    bboxes_idx = []
                    ratios = []

                    for box_idx in range(json_info['predictions']):
                        # get coordinates in original image that's why r*coord
                        bbox = [int(r*coord) for coord in json_info['box_' + str(box_idx)]['pred']]

                        # make sure we're not going over the image limits
                        bbox[1] = max(0, bbox[1])
                        bbox[3] = min(page_img.shape[0], bbox[3])
                        bbox[0] = max(0,bbox[0])
                        bbox[2] = min(page_img.shape[1], bbox[2])

                        area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
                        if area < 100:
                            continue

                        word_img = page_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                        bboxes += [bbox]
                        bboxes_idx += ['box_' + str(box_idx)]
                        
                        # save ratio of x/y of image
                        ratios += [(bbox[2]-bbox[0])/float(bbox[3] - bbox[1])]

                        word_imgs += [word_img]
                        # if params.debug:
                            # word = Image.fromarray(word_img)
                            # word.show()

                    # save information about where each image is from, and its bounding box info
                    candidates_labels += [{'page': os.path.join(collection, page_name),
                                           'bbox': bbox,
                                           'bbox_idx': bboxes_idx[idx],
                                           'xy_ratio': ratios[idx]} for idx, bbox in enumerate(bboxes)]

                    # get phoc representation of words
                    word_imgs = [torch.autograd.Variable(torch.from_numpy(word_img)).float() for word_img in word_imgs]

                    if params.gpu_id is not None:
                        word_imgs = [word_img.cuda(params.gpu_id[0]) for word_img in word_imgs]

                    word_imgs = [word_img.unsqueeze(0) for word_img in word_imgs]
                    word_imgs = [word_img.unsqueeze(0) for word_img in word_imgs]

                    # get phoc representation of words
                    word_phocs = []
                    for word_img in word_imgs:
                        word_phocs += [torch.sigmoid(cnn(word_img))]

                    word_phocs = [word_phoc.data.cpu().numpy().flatten() for word_phoc in word_phocs]
                    word_phocs = np.vstack(word_phocs)
                    # concatenate along the row dimension as a numpy array
                    # if very first item
                    if coll_idx == 0 and page_idx == 0:
                        candidates = np.vstack(word_phocs)
                    else:
                        candidates = np.vstack((candidates, word_phocs))
                    
                    t.update(1)
    else:
        folders = os.listdir(path_to_cands)
        folders = [folder for folder in folders if folder != '.DS_Store']

        if params.debug:
            folders = folders[:2]

            for i, folder in enumerate(folders):
                word_img_names = sorted(
                    [elem for elem in os.listdir(os.path.join(path_to_cands, folder)) if elem.endswith('.jpg')])

                candidates_labels += ['_'.join([folder, word_img_name[:-4]]) for word_img_name in word_img_names]
                word_imgs = [img_io.imread(os.path.join(path_to_cands, folder, word_img_name))
                             for word_img_name in word_img_names]

                word_imgs = [torch.autograd.Variable(torch.from_numpy(word_img)).float() for word_img in word_imgs]

                if params.gpu_id is not None:
                    word_imgs = [word_img.cuda(params.gpu_id[0]) for word_img in word_imgs]

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
    np.save(os.path.join(params.model_dir,'candidates.npy'), candidates)

    # save candidates_labels as json file
    with open(os.path.join(params.model_dir, 'candidates_labels.json'), 'w') as filehandle:
        json.dump(candidates_labels, filehandle)

    return candidates, candidates_labels

if __name__ == '__main__':
    args = parser.parse_args()
    args.num_save_im = int(args.num_save_im)

    # if GPU is not available, use CPU
    if not torch.cuda.is_available():
        args.gpu_id = None

    utils.set_logger(os.path.join(args.model_dir, 'eval.log'))

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # merge params and args into one object
    params.args_to_params(args)
    eval(params)
