#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys
import pickle
import argparse
import util

W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red


def predict(gmms, test_frame):
    scores = []
    for gmm_name, gmm in gmms.items():
        scores.append((gmm_name, gmm.score(test_frame)))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', dest='index_file', default='testing_set.idx', help= 'path to the index file')
    parser.add_argument('-m','--model', dest='model_dir', default='models', help= 'directiory to store models')
    args = parser.parse_args()
    if not os.path.exists(args.index_file):
        print('{} not found.'.format(args.index_file))
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.model_dir):
        print('{} not found.'.format(args.model_dir))
        parser.print_help()
        sys.exit(1)
    return args

def main():
    args = parse_args()
    gmms = {}
    print('loading GMM models...')
    for filename in glob.glob(os.path.join(args.model_dir, '*.gmm')):
        name = os.path.splitext(os.path.basename(filename))[0]
        gmms[name] = pickle.load(open(filename,'rb'))
        print('name={}, GMM={}'.format(name,filename))
    print('\r\nloading wav files for testing...')
    correct = 0
    total=0
    with open(args.index_file,'r') as fp:
        for line in fp:
            items = line.split(' ')
            if len(items) < 2:
                continue
            name = items[1].rstrip()
            result = predict(gmms, util.load_features(items[0]))
            if name == result[0][0]:
                print('Ground Truth: %s: Predict: %s' % (name, ' / '.join(map(lambda x: '%s = %f' % x, result[:3]))))
                correct += 1
            else:
                str = 'Ground Truth: %s: Predict: %s' % (name, ' / '.join(map(lambda x: '%s = %f' % x, result[:3])))
                print(R + str + W)
            total +=1
    print('\r\nOverall Accuracy: %f%%' % (100*float(correct) / total))

if __name__ == '__main__':
    main()
