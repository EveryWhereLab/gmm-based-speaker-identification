#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sklearn.mixture
import sys
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy import linalg
import argparse
from random import seed
from random import randint
import pickle
import util

SAVE_MODEL=True

def bic_fit_gmm(X,speaker, output_dir):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 20)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    best_n_components = 1
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = sklearn.mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type, max_iter=150)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_n_components = n_components
                best_gmm = gmm
    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
    clf = best_gmm
    print('Best GMM:')
    print(clf)
    bars = []
    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    splot.axis('equal')
    Y_ = clf.predict(X)
    # Select two dimensions to draw a scatter plot.
    maximum_values = np.amax(clf.means_,0)
    minimum_values = np.amin(clf.means_,0)
    mean_diff = maximum_values - minimum_values
    sort_idx = np.argsort(mean_diff)
    idx1=sort_idx[-1]
    idx2=sort_idx[-2]
    colors = cm.rainbow(np.linspace(0, 1, clf.n_components))
    for i, mean in enumerate(clf.means_):
        subcov = np.zeros((2, 2))
        if clf.covariance_type == 'full':
            subcov[0,0] = clf.covariances_[i,idx1,idx1]
            subcov[0,1] = clf.covariances_[i,idx1,idx2]
            subcov[1,0] = clf.covariances_[i,idx2,idx1]
            subcov[1,1] = clf.covariances_[i,idx2,idx2]
        elif clf.covariance_type == 'tied':
            subcov[0,0] = clf.covariances_[idx1,idx1]
            subcov[0,1] = clf.covariances_[idx1,idx2]
            subcov[1,0] = clf.covariances_[idx2,idx1]
            subcov[1,1] = clf.covariances_[idx2,idx2]
        elif clf.covariance_type == 'diag':
            subcov[0,0] = clf.covariances_[i,idx1]
            subcov[1,1] = clf.covariances_[i,idx2]
        elif clf.covariance_type == 'spherical':
            subcov[0,0] = clf.covariances_[i]
            subcov[1,1] = clf.covariances_[i]
        v, w = linalg.eigh(subcov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, idx1], X[Y_ == i, idx2], .8, color=colors[i])
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse((mean[idx1],mean[idx2]), v[0], v[1], 180. + angle, color=colors[i])
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: {} model, {} components, x:idx {}, y:idx {}'.format(clf.covariance_type, clf.n_components,idx1,idx2))
    plt.subplots_adjust(hspace=.35, bottom=.02)
    filename = speaker + '.png'
    output = os.path.join(output_dir,filename)
    plt.savefig(output)
    return clf

def bic_search_gmm_hyperparameters(voices, speaker, output_dir='out', n_components=16, dropout_percentage=20):
    feats = None
    for i in range(len(voices)):
        voice = voices[i]
        if randint(0, 100) > dropout_percentage:
            feat = util.load_features(voice)
            if feats is None:
                feats = np.empty((0, feat.shape[1]))
            feats = np.vstack([feats, feat])
    if feats is not None:
        return bic_fit_gmm(feats, speaker, output_dir)
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', dest='index_file', default='training_set.idx', help= 'path to the index file')
    parser.add_argument('-o','--output', dest='output_dir', default='out', help= 'dir to put scatter plots')
    parser.add_argument('-m','--model', dest='model_dir', default='models', help= 'directiory to store models')
    args = parser.parse_args()
    if not os.path.exists(args.index_file):
        print('{} not found.'.format(args.index_file))
        parser.print_help()
        sys.exit(1)
    return args 

def main():
    seed(1)
    args = parse_args()
    speaker_sounds = util.load_sounds_meta_data(args.index_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if SAVE_MODEL is True and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    for speaker in speaker_sounds:
        print('Searching GMM hyperparameters for speaker \"{}\" using BIC...'.format(speaker))
        estimator = bic_search_gmm_hyperparameters(speaker_sounds[speaker], speaker, args.output_dir)
        if SAVE_MODEL is True:
            filename =  os.path.join(args.model_dir, os.path.basename(speaker) +'.gmm')
            pickle.dump(estimator,open(filename,'wb'))
            print('Model saved in path:'+filename)
    print('\nBIC searhing is complete.')

if __name__ == '__main__':
    main()
