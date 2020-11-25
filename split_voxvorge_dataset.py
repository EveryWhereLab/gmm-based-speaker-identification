#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys
import argparse
from random import seed
from random import randint

TESTING_PERCENTAGE=20
TRAINING_SET_IDX_FILE = "training_set.idx"
TESTING_SET_IDX_FILE = "testing_set.idx"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', dest='path', default='extracted', help= 'the directory containing the voxforge dataset')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        print('{} not found.'.format(args.path))
        parser.print_help()
        sys.exit(1)
    return args

def main():
    args = parse_args()
    # seed random number generator
    seed(1)
    with open(TRAINING_SET_IDX_FILE,'w') as training, open(TESTING_SET_IDX_FILE, 'w') as testing:
        for root, dirs, files in os.walk(args.path):
            for dir in dirs:
                readme_path = os.path.join(root, dir,'etc/README')
                if os.path.isfile(readme_path):
                    with open(readme_path) as fp:
                        line = fp.readline()
                        items = line.rstrip().split(':')
                        if len(items) > 1:
                            voices = glob.glob(os.path.join(root, dir,'wav/*.wav'))
                            for voice in voices:
                                if randint(0, 100) > TESTING_PERCENTAGE:
                                    training.write('{} {}\n'.format(voice, items[1]))
                                else:
                                    testing.write('{} {}\n'.format(voice, items[1]))
        print('\"{}\" and \"{}\" files are successfully generated.'.format(TRAINING_SET_IDX_FILE, TESTING_SET_IDX_FILE))

if __name__ == '__main__':
    main()

