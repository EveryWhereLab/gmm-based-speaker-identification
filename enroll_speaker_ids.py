#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sklearn.mixture
import sys
import pickle
import argparse
import util

N_COMPONENTS=8
#N_COMPONENTS=10

def train_from_voices(voices, n_components=16):
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components,max_iter=150)
    feats = None
    for i in range(len(voices)):
        voice = voices[i]
        feat = util.load_features(voice)
        if feats is None:
            feats = np.empty((0, feat.shape[1]))
        feats = np.vstack([feats, feat])
    gmm.fit(feats)
    return gmm

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
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    for speaker in speaker_sounds:
        print('Training speaker {}\'s GMM model...'.format(speaker))
        gmm = train_from_voices(speaker_sounds[speaker],n_components=N_COMPONENTS)
        #save the model to disk
        filename =  os.path.join(args.model_dir, os.path.basename(speaker) +'.gmm')
        pickle.dump(gmm,open(filename,'wb'))
        print('Model saved in path:'+filename)
    print('\nTraining is complete.')

if __name__ == '__main__':
    main()
