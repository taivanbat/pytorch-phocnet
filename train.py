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

def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

def train(params, args):
    logging.info('--- Running PHOCNet Training ---')

    # sanity checks
    if not torch.cuda.is_available():
        logging.warning('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    # print out the used arguments
    """
    logging.info('###########################################')
    logging.info('Experiment Parameters:')
    for key, value in vars(args).iteritems():
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
                              fixed_image_size=args.fixed_image_size,
                              min_image_width_height=params.min_image_width_height)

    if params.dataset == 'wiener':
        train_set = WienerDataset(wiener_root_dir='data/wiener',
                                  embedding=params.embedding_type,
                                  phoc_unigram_levels=params.phoc_unigram_levels,
                                  fixed_image_size=args.fixed_image_size,
                                  min_image_width_height=params.min_image_width_height)

    if params.dataset == 'iam':
        train_set = IAMDataset(gw_root_dir='../../../phocnet-pytorch-master/data/IAM',
                               image_extension='.png',
                               embedding=params.embedding_type,
                               phoc_unigram_levels=params.phoc_unigram_levels,
                               fixed_image_size=args.fixed_image_size,
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

    cnn = PHOCNet(n_out=train_set[0][1].shape[0],
                  input_channels=1,
                  gpp_type='gpp',
                  pooling_levels=([1], [5]))

    cnn.init_weights()

    if args.restore_file is not None:
        #cnn.load_state_dict(torch.load('PHOCNet.pt', map_location=lambda storage, loc: storage))
        my_torch_load(cnn, os.path.join(args.model_dir, args.restore_file))

    loss_selection = params.loss_selection # 'BCE' or 'cosine'
    if loss_selection == 'BCE':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        loss = CosineLoss(size_average=False, use_sigmoid=True)
    else:
        raise ValueError('not supported loss function')

    # move CNN to GPU
    if args.gpu_id is not None:
        if len(args.gpu_id) > 1:
            cnn = nn.DataParallel(cnn, device_ids=args.gpu_id)
            cnn.cuda()
        else:
            cnn.cuda(args.gpu_id[0])

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
    # TODO: step setting best_mAP manually
    best_val_acc = 0.0
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
            
                if args.gpu_id is not None:
                    if len(args.gpu_id) > 1:
                        word_img = word_img.cuda()
                        embedding = embedding.cuda()
                    else:
                        word_img = word_img.cuda(args.gpu_id[0])
                        embedding = embedding.cuda(args.gpu_id[0])

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
            logging.info('Evaluating net after %d epochs', epoch + 1)
            val_acc = evaluate_cnn(cnn=cnn,
                         dataset_loader=test_loader,
                         args=args)

            is_best = val_acc >= best_val_acc

            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(args.model_dir, "metrics_val_best_weights.json")

                # TODO: replace this with more general case
                val_metrics = {}
                val_metrics['mAP'] = best_val_acc

                utils.save_dict_to_json(val_metrics, best_json_path)
                my_torch_save(cnn, os.path.join(args.model_dir, 'PHOCNET_best.pt'))
    my_torch_save(cnn, os.path.join(args.model_dir, 'PHOCNET_last.pt'))

def evaluate_cnn(cnn, dataset_loader, args):
    # set the CNN in eval mode
    cnn.eval()
    logging.info('Computing net output:')
    qry_ids = [] #np.zeros(len(dataset_loader), dtype=np.int32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm(dataset_loader)):
        if args.gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda(args.gpu_id[0])
            embedding = embedding.cuda(args.gpu_id[0])
            #word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)
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

    # TODO make it so that we can choose the evaluation metric from args 
    ave_precs_qbe = map_from_query_test_feature_matrices(query_features = qry_outputs,
                                                         test_features=outputs,
                                                         query_labels = qry_class_ids,
                                                         test_labels=class_ids,
                                                         metric='cosine',
                                                         drop_first=True)
    
    mAP = np.mean(ave_precs_qbe[ave_precs_qbe > 0])
    logging.info('mAP: %3.2f', mAP*100)

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
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    train(params, args)
