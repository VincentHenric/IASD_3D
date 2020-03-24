#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:55:06 2020

@author: henric
"""

import os

import numpy as np

PATH = '../data'
CLASSES = {1: 'man-made terrain',
           2: 'natural terrain',
           3: 'high vegetation',
           4: 'low vegetation',
           5: 'buildings',
           6: 'hard scape',
           7: 'scanning artefacts',
           8: 'cars'}

def load_dataset(filename, max_rows=None):
    data = np.loadtxt(os.path.join(PATH, filename), max_rows=max_rows)
    return data
    #return data[:,:3], data[:,[3]], data[:,4:]

def load_labels(filename, max_rows=None):
    labels = np.loadtxt(os.path.join(PATH, filename), max_rows=max_rows)
    return labels

data = load_dataset('bildstein_station1_xyz_intensity_rgb.txt', max_rows = 100)
labels = load_labels('bildstein_station1_xyz_intensity_rgb.labels', max_rows = 100)
    