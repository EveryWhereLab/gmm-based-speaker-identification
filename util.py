#!/usr/bin/env python
# -*- coding: utf-8 -*-

import librosa

#LIFTERNUMBER=50
LIFTERNUMBER=0

def load_features(audio_path):
    y, sr = librosa.load(audio_path)
    y_trim = librosa.effects.remix(y, intervals=librosa.effects.split(y))
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, lifter=LIFTERNUMBER)
    return mfcc.T

def load_sounds_meta_data(index_file):
    speaker_sounds = {}
    with open(index_file,'r') as fp:
        for line in fp:
            line = line.strip()
            items = line.split(' ')
            if len(items) < 2:
                continue
            name = items[1].strip()
            original_list = speaker_sounds.get(name)
            if original_list is None:
                speaker_sounds[name]=[items[0]]
            else:
                original_list.append(items[0])
                speaker_sounds[name] = original_list
    return speaker_sounds
