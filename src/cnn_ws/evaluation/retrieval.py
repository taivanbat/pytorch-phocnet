'''
Created on Dec 1, 2014

@author: ssudholt
'''
import os

import numpy as np
from PIL import Image
from skimage import io as img_io

from scipy.spatial.distance import cdist, pdist, squareform
from src.cnn_ws.string_embeddings.phoc import build_phoc_descriptor, get_unigrams_from_strings

def average_precision(ret_vec_relevance, gt_relevance_num=None):
    '''
    Computes the average precision from a list of relevance items
    
    Params:
        ret_vec_relevance: A 1-D numpy array containing ground truth (gt)
            relevance values
        gt_relevance_num: Number of relevant items in the data set
            (with respect to the ground truth)
            If None, the average precision is calculated wrt the number of
            relevant items in the retrieval list (ret_vec_relevance)

    Returns:
        The average precision for the given relevance vector.
    '''
    if ret_vec_relevance.ndim != 1:
        raise ValueError('Invalid ret_vec_relevance shape')

    ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
    ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
    ret_vec_precision = ret_vec_cumsum / ret_vec_range

    if gt_relevance_num is None:
        n_relevance = ret_vec_relevance.sum()
    else:
        n_relevance = gt_relevance_num

    if n_relevance > 0:
        ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
    else:
        ret_vec_ap = 0.0
    return ret_vec_ap

def map_from_query_test_feature_matrices(query_features, test_features,
                                         query_labels, test_labels,
                                         metric,
                                         drop_first=False):
    '''
    Compute the mAP for a given list of queries and test instances
    Each query is used to rank the test samples
    :param query_features: (2D ndarray)
        feature representation of the queries
    :param test_features: (2D ndarray)
        feature representation of the test instances
    :param query_labels: (1D ndarray or list)
        the labels corresponding to the queries (either numeric or characters)
    :param test_labels: (1D ndarray or list)
        the labels corresponding to the test instances (either numeric or characters)
    :param metric: (string)
        the metric to be used in calculating the mAP
    :param drop_first: (bool)
        whether to drop the first retrieval result or not
    '''
    # some argument error checking
    if query_features.shape[1] != test_features.shape[1]:
        raise ValueError('Shape mismatch')
    if query_features.shape[0] != len(query_labels):
        raise ValueError('The number of query feature vectors and query labels does not match')
    if test_features.shape[0] != len(test_labels):
        raise ValueError('The number of test feature vectors and test labels does not match')

    # compute the nearest neighbors
    dist_mat = cdist(XA=query_features, XB=test_features, metric=metric)
    retrieval_indices = np.argsort(dist_mat, axis=1)

    # create the retrieval matrix
    retr_mat = np.tile(test_labels, (len(query_labels), 1))
    row_selector = np.transpose(np.tile(np.arange(len(query_labels)), (len(test_labels), 1)))
    retr_mat = retr_mat[row_selector, retrieval_indices]

    # create the relevance matrix
    relevance_matrix = retr_mat == np.atleast_2d(query_labels).T
    if drop_first:
        relevance_matrix = relevance_matrix[:, 1:]

    # calculate mAP and APs
    avg_precs = np.array([average_precision(row) for row in relevance_matrix], ndmin=2).flatten()
    mean_ap = np.mean(avg_precs)
    return mean_ap, avg_precs

def map_from_feature_matrix(features, labels, metric, drop_first):
    '''
    Computes mAP and APs from a given matrix of feature vectors
    Each sample is used as a query once and all the other samples are
    used for testing. The user can specify whether he wants to include
    the query in the test results as well or not.

    :param features:(2D ndarray)
        the feature representation from which to compute the mAP
    :param labels: (1D ndarray or list)
        the labels corresponding to the features (either numeric or characters)
    :param metric: (string)
        the metric to be used in calculating the mAP
    :param drop_first: (bool)
        whether to drop the first retrieval result or not
    '''
    # argument error checks
    if features.shape[0] != len(labels):
        raise ValueError('The number of feature vectors and number of labels must match')
    # compute the pairwise distances from the features
    dist_mat = squareform(pdist(X=features, metric=metric))
    np.fill_diagonal(dist_mat, -1)  # make sure identical indices are sorted to the front
    inds = np.argsort(dist_mat, axis=1)
    retr_mat = np.tile(labels, (features.shape[0], 1))

    # compute two matrices for selecting rows and columns
    # from the label matrix
    # -> advanced indexing
    row_selector = np.transpose(np.tile(np.arange(features.shape[0]), (features.shape[0], 1)))
    retr_mat = retr_mat[row_selector, inds]

    # create the relevance matrix
    rel_matrix = retr_mat == np.atleast_2d(labels).T
    if drop_first:
        rel_matrix = rel_matrix[:, 1:]

    # calculate mAP and APs
    avg_precs = np.array([average_precision(row) for row in rel_matrix])
    mean_ap = np.mean(avg_precs)
    return mean_ap, avg_precs

def run_query(num_save, candidates, candidates_labels, queries, params, wiener=True):
    '''
    inputs:
        num_save -- integer indicating how many results to return, -1 means return entire pool of candidates, ranked
        candidates -- an NxM matrix where each row is the estimated PHOC of the word image
        queries -- the list of words that we would like to find. We also convert this to an LxM matrix where each
            row l is the PHOC representation of the l-th word

    :return: indices of n highest search results as a list
    '''
    unigrams = []
    if wiener:
        # TODO create a text file with all these unigrams!!!!!!!!!!!!!!!, so we don't have to create it
        # again and again every time we evaluate or train.

        # QUESTION: how are we going to handle characters that are not in the training set ?
        wiener_root_dir = 'data/wiener'
        phoc_unigram_levels = (1, 2, 4, 8)
        img_filenames = sorted([elem for elem in os.listdir(os.path.join(wiener_root_dir, 'queries', 'word_images'))
                                if elem.endswith('.png')])
        all_names = ''.join(img_filenames)
        all_names = all_names.decode('utf-8')
        unigrams = list(set(all_names))
        unigrams += [chr(i) for i in range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
        unigrams = list(set(unigrams))

    query_phoc = build_phoc_descriptor(words=queries,
                                       phoc_unigrams=unigrams,
                                       unigram_levels=phoc_unigram_levels)

    dist = cdist(query_phoc, candidates, params.eval_metric)
    sorted_results = np.argsort(dist, axis=1)
    """   
    # TODO: filter out images that don't have similar ratios
    num_ratio_matches= 0 

    if params.save_im_results:
        # display the word along with its top 10 results
        for j, query in enumerate(queries):
            # make directory with query name
            default_dir = os.path.join(params.model_dir, 'results')
            results_dir = os.path.join(default_dir, query)
            os.makedirs(results_dir)

            # save top 10 results
            for result_i in range(num_save):
                idx = sorted_results[j, result_i]
                im_label = candidates_labels[idx].split('_')
                im_path = os.path.join(wiener_root_dir, 'candidates', 'word_images', im_label[0], im_label[1] + '.jpg')
                img = img_io.imread(im_path)
                img = Image.fromarray(np.uint8(img * 255))
                img.save(os.path.join(results_dir, str(result_i) + '_' + candidates_labels[idx] + '.jpg'))
    """
    return sorted_results
