#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sklearn.mixture
import sys
from sklearn.model_selection import GridSearchCV
import pickle
import argparse
import util

SAVE_MODEL=True

def grid_search_gmm_hyperparameters(voices, n_components=16):
    feats = None
    for i in range(len(voices)):
        voice = voices[i]
        feat = util.load_features(voice)
        if feats is None:
            feats = np.empty((0, feat.shape[1]))
        feats = np.vstack([feats, feat])
    tuned_parameters = {'n_components': range(3, 20), 'covariance_type': ['tied','full','spherical'], 'init_params': ['kmeans', 'random'], 'max_iter': [150]}
    #construct grid search object that uses 3 fold cross validation
    clf = GridSearchCV(sklearn.mixture.GaussianMixture(),tuned_parameters,cv=3) 
    #fit the data
    clf.fit(feats)
    print('Best hyperparameters:')
    print(clf.best_estimator_)
    return clf.best_estimator_

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', dest='index_file', default='training_set.idx', help= 'path to the index file')
    parser.add_argument('-m','--model', dest='model_dir', default='models', help= 'directiory to store models')
    args = parser.parse_args()
    if not os.path.exists(args.index_file):
        print('{} not found.'.format(args.index_file))
        parser.print_help()
        sys.exit(1)
    return args 

def main():
    args = parse_args()
    speaker_sounds = util.load_sounds_meta_data(args.index_file)
    if SAVE_MODEL is True and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    for speaker in speaker_sounds:
        print('Searching GMM hyperparameters for speaker \"{}\" using grid search...'.format(speaker))
        estimator = grid_search_gmm_hyperparameters(speaker_sounds[speaker])
        if SAVE_MODEL is True:
            filename =  os.path.join(args.model_dir, os.path.basename(speaker) +'.gmm')
            pickle.dump(estimator,open(filename,'wb'))
            print('Model saved in path:'+filename)
    print('\nGrid searhing is complete.')

if __name__ == '__main__':
    main()
