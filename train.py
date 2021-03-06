'''
Created on Sep 3, 2017

@author: georgeretsi

do not support multi-gpu yet. needs thread manipulation
- works on GW + IAM
- new way to load dataset
- augmentation with dataloader
- Hardcoded selections (augmentation - default:YES, load pretrained model with hardcoded name...
- do not normalize with respect to iter size (or batch size) for speed
- add fixed size selection (not hardcoded)
- save and load hardcoded name 'PHOCNet.pt'

edited by @taivanbat for project
'''
import argparse
import logging
import sys
import os

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json 

import copy
from experiments.cnn_ws_experiments.datasets.iam_alt import IAMDataset
from experiments.cnn_ws_experiments.datasets.gw_alt import GWDataset
from experiments.cnn_ws_experiments.datasets.wiener_alt import WienerDataset

#from cnn_ws.transformations.homography_augmentation import HomographyAugmentation
from src.cnn_ws.losses.cosine_loss import CosineLoss

from src.cnn_ws.models.myphocnet import PHOCNet
from src.cnn_ws.evaluation.retrieval import map_from_feature_matrix, map_from_query_test_feature_matrices
from torch.utils.data.dataloader import _DataLoaderIter
from torch.utils.data.sampler import WeightedRandomSampler

from src.cnn_ws.utils.save_load import my_torch_save, my_torch_load
from src.cnn_ws.utils import utils

parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--model_dir', default='models/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--gpu_id', '-gpu', action='store',
                    type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                    default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--fixed_image_size', '-fim', action='store',
                    type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                    default=None,
                    help='Specifies the images to be resized to a fixed size when presented to the CNN. Argument must be two comma seperated numbers.')
parser.add_argument('--load_best_metrics', action='store_true', help='If we want to load best metrics')
parser.add_argument('--dont_save_model', action='store_true', help='If we want to load best metrics')

def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

def train(params):
    logging.info('--- Running PHOCNet Training ---')

    # sanity checks
    if not torch.cuda.is_available():
        logging.warning('Could not find CUDA environment, using CPU mode')
        params.gpu_id = None

    # print out the used arguments
    """
    logging.info('###########################################')
    logging.info('Experiment Parameters:')
    for key, value in vars(params).iteritems():
        logging.info('%s: %s', str(key), str(value))
    logging.info('###########################################')
    """


    # prepare datset loader
    #TODO: add augmentation
    logging.info('Loading dataset %s...', params.dataset)
    if params.dataset == 'gw':
        train_set = GWDataset(gw_root_dir='../../data/gw',
                              cv_split_method='almazan',
                              cv_split_idx=1,
                              image_extension='.tif',
                              embedding=params.embedding_type,
                              phoc_unigram_levels=params.phoc_unigram_levels,
                              fixed_image_size=params.fixed_image_size,
                              min_image_width_height=params.min_image_width_height)

    if params.dataset == 'wiener':
        train_set = WienerDataset(wiener_root_dir='data/wiener',
                                  embedding=params.embedding_type,
                                  phoc_unigram_levels=params.phoc_unigram_levels,
                                  fixed_image_size=params.fixed_image_size,
                                  min_image_width_height=params.min_image_width_height)

    if params.dataset == 'iam':
        train_set = IAMDataset(iam_root_dir='data/iam',
                               image_extension='.png',
                               embedding=params.embedding_type,
                               phoc_unigram_levels=params.phoc_unigram_levels,
                               fixed_image_size=params.fixed_image_size,
                               min_image_width_height=params.min_image_width_height)

    test_set = copy.copy(train_set)

    train_set.mainLoader(partition='train')
    test_set.mainLoader(partition='test', transforms=None)

    # augmentation using data sampler
    n_train_images = 500000
    augmentation = False

    if augmentation:
        train_loader = DataLoader(train_set,
                                  sampler=WeightedRandomSampler(train_set.weights, n_train_images),
                                  batch_size=params.batch_size,
                                  num_workers=4)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=params.batch_size, shuffle=True,
                                  num_workers=4)

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)
    # load CNN
    logging.info('Preparing PHOCNet...')

    # GPP vs SPP 
    cnn = PHOCNet(n_out=train_set[0][1].shape[0],
                  input_channels=1,
                  gpp_type=params.gpp_type,
                  pooling_levels=params.pooling_levels)

    cnn.init_weights()

    if params.restore_file is not None:
        my_torch_load(cnn, params.restore_file)

    loss_selection = params.loss_selection # 'BCE' or 'cosine'
    if loss_selection == 'BCE':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        loss = CosineLoss(size_average=False, use_sigmoid=True)
    else:
        raise ValueError('not supported loss function')

    # move CNN to GPU
    if params.gpu_id is not None:
        if len(params.gpu_id) > 1:
            cnn = nn.DataParallel(cnn, device_ids=params.gpu_id)
            cnn.cuda()
        else:
            cnn.cuda()

    # run training
    lr_cnt = 0
    params.learning_rate_step = learning_rate_step_parser(params.learning_rate_step)
    max_iters = params.learning_rate_step[-1][0]
    if params.solver_type == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(), params.learning_rate_step[0][1],
                                    momentum=params.momentum,
                                    weight_decay=params.weight_decay)

    if params.solver_type == 'Adam':
        optimizer = torch.optim.Adam(cnn.parameters(), params.learning_rate_step[0][1],
                                    weight_decay=params.weight_decay)

    # set best_mAP, because loading pretrained model that we shouldn't overwrite
    best_val_acc = params.best_mAP
    optimizer.zero_grad()
    logging.info('Training:')

    loss_avg = utils.RunningAverage()
    
    for epoch in range(params.epochs):
        logging.info('epoch: ' + str(epoch))
        with tqdm(total=len(train_loader)) as t:
            for iter_idx, sample in enumerate(train_loader):
                word_img, embedding, _, _ = sample

                # get actual step number 
                current_step = epoch*len(train_loader) + iter_idx 
            
                if params.gpu_id is not None:
                    if len(params.gpu_id) > 1:
                        word_img = word_img.cuda()
                        embedding = embedding.cuda()
                    else:
                        word_img = word_img.cuda()
                        embedding = embedding.cuda()

                word_img = torch.autograd.Variable(word_img).float()
                embedding = torch.autograd.Variable(embedding).float()
                output = cnn(word_img)
                ''' BCEloss ??? '''
                loss_val = loss(output, embedding)*params.batch_size
                loss_val.backward()

                optimizer.step()
                optimizer.zero_grad()
                
                # change lr
                if (current_step + 1) == params.learning_rate_step[lr_cnt][0] and (current_step+1) != max_iters:
                    logging.info('changing learning rate at step %d from %f to %f', current_step + 1, 
                                 params.learning_rate_step[lr_cnt][1], params.learning_rate_step[lr_cnt + 1][1])
                    lr_cnt += 1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = params.learning_rate_step[lr_cnt][1]

                # update the average loss
                loss_avg.update(loss_val.item())
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()
        
        if epoch % params.test_at_epoch == 0: # and iter_idx > 0:
            logging.info('Evaluating at epoch %d', epoch + 1)
            val_acc = evaluate_cnn(cnn=cnn,
                         dataset_loader=test_loader,
                         params=params)
                
            logging.info('mAP: %3.2f', val_acc*100) 
            logging.info('loss: {:05.3f}'.format(loss_avg()))
            logging.info('-----------------------------------')

            is_best = val_acc >= best_val_acc

            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(params.model_dir, "metrics_val_best_weights.json")

                # TODO: replace this with more general case
                val_metrics = {}
                val_metrics['mAP'] = best_val_acc
                val_metrics['loss'] = loss_avg() 

                utils.save_dict_to_json(val_metrics, best_json_path)
                if not params.dont_save_model:
                    my_torch_save(cnn, os.path.join(params.model_dir, 'PHOCNET_best.pt'))
    
    if not params.dont_save_model: 
        my_torch_save(cnn, os.path.join(params.model_dir, 'PHOCNET_last.pt'))

def evaluate_cnn(cnn, dataset_loader, params):
    # set the CNN in eval mode
    cnn.eval()
    logging.info('Computing net output:')
    qry_ids = [] #np.zeros(len(dataset_loader), dtype=np.int32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm(dataset_loader)):
        if params.gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda()
            embedding = embedding.cuda()
            #word_img, embedding = word_img.cuda(params.gpu_id), embedding.cuda(params.gpu_id)
        word_img = torch.autograd.Variable(word_img).float()
        embedding = torch.autograd.Variable(embedding).float()
        ''' BCEloss ??? '''
        output = torch.sigmoid(cnn(word_img))
        #output = cnn(word_img)
        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0,0]
        if is_query[0] == 1:
            qry_ids.append(sample_idx)  #[sample_idx] = is_query[0]

    '''
    # find queriesl

    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]

    # remove stopwords if needed
    
    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logging.info('Computing mAPs...')

    # we choose the evaluation metric from the given parameters 
    ave_precs_qbe = map_from_query_test_feature_matrices(query_features = qry_outputs,
                                                         test_features=outputs,
                                                         query_labels = qry_class_ids,
                                                         test_labels=class_ids,
                                                         metric=params.eval_metric,
                                                         drop_first=True)
    
    mAP = np.mean(ave_precs_qbe[ave_precs_qbe > 0])
    # clean up -> set CNN in train mode again
    cnn.train()

    # return mAP and save if new best accuracy
    return mAP

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    json_path = os.path.join(args.model_dir, 'params.json')
    print(json_path)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    params = utils.Params(json_path)
    params.args_to_params(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id[0]) 

    # load best accuracy into params
    if params.restore_file is not None and params.load_best_metrics:
        with open(os.path.join(params.model_dir, 'metrics_val_best_weights.json')) as f:
            best_metrics = json.load(f)
        params.best_mAP = best_metrics["mAP"]
    else:
        params.best_mAP = 0
    
    train(params)
