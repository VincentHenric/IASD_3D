#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:55:06 2020

@author: henric
"""

import os

from utils.ply import write_ply, read_ply
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

ZOOM_COORDINATES = {
        'bildstein1':{'min_x':12, 'max_x':55, 'min_y':-10, 'max_y':40},
        'bildstein3':{'min_x':-20, 'max_x':17, 'min_y':-70, 'max_y':-25},
        'bildstein5':{'min_x':-40, 'max_x':-5, 'min_y':-35, 'max_y':15}
        }

REF_POINTS = [{'bildstein1':{'x':30.249, 'y':32.927, 'z':31.1},
               'bildstein3':{'x':3.597, 'y':-65.583, 'z':28.228},
               'bildstein5':{'x':-30.251, 'y':3.979, 'z':32.137}},
              {'bildstein1':{'x':46.177, 'y':26.846, 'z':30.41},
               'bildstein3':{'x':-13.306, 'y':-62.783, 'z':28.636},
               'bildstein5':{'x':-14.469, 'y':-2.091, 'z':32.137}}
]
#np.array([[3.16, 4.73, 1.12],
#        [24.31, 51.87, -3.54],
#        [47.4, 63.9, -1.15],
#        [48.6, -27.3, 6.46]])
    
def load_dataset(filename, max_rows=None):
    data = np.loadtxt(os.path.join(PATH, filename), max_rows=max_rows)
    return data
    #return data[:,:3], data[:,[3]], data[:,4:]

def save_dataset_npy(data, filename_rad):
    np.save(os.path.join(PATH, 'transformed_{}.npy'.format(filename_rad)), data)
    #np.savetxt(os.path.join(PATH, '{}_transformed.txt'.format(filename_rad)), data)

def save_dataset_ply(data, filename_rad, prefix='transformed'):
    if prefix!='':
        prefix = prefix + '_'
    write_ply(os.path.join(PATH, '{}{}.ply'.format(prefix, filename_rad)),
              data,
              ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue'])

def normalize_intensities(intensities, low=-2048, high=2047):
    """
    apply linear transformation to intensities to make them fit in 0-255 range
    """
    intensities = (intensities - low)/(high-low)*255
    return intensities

def clean_dataset(filename_rad, label=5, max_rows=None):
    """
    filter in only a given label
    """
    data = load_dataset(filename_rad+'.txt', max_rows)
    labels = load_dataset(filename_rad+'.labels', max_rows)
    data = np.concatenate((data, labels[:,None]), axis=1)
    data = data[data[:,-1]==label]
    data = np.delete(data, -1, axis=1)
    save_dataset_ply(data, filename_rad)
    
def select_points(filename_rad):
    """
    select points based on coordinates from ZOOM_COORDINATES
    """
    cloud_ply = read_ply(os.path.join(PATH, 'transformed_' + filename_rad + '.ply'))
    
    for coord in ['x', 'y', 'z']:
        min_val = ZOOM_COORDINATES[filename_rad].get('min_'+coord)
        if min_val:
            cloud_ply = cloud_ply[cloud_ply[coord]>=min_val]
            
        max_val = ZOOM_COORDINATES[filename_rad].get('max_'+coord)
        if min_val:
            cloud_ply = cloud_ply[cloud_ply[coord]<=max_val]
            
    return cloud_ply
    
def center(cloud_points):
    return cloud_points - cloud_points.mean(axis=0)

if __name__ == '__main__':
    if False:
        filename_rad = 'bildstein1'
        clean_dataset(filename_rad)
        
    if True:
        filename_rad = 'bildstein1'
        
        # selection
        cloud_ply = select_points(filename_rad)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        
        # processing
        #cloud_points = center(cloud_points)
        intensities = normalize_intensities(intensities, low=-2048, high=2047)
        
        save_dataset_ply((cloud_points, intensities[:,None].astype('uint8'), colors.astype('uint8')),
                       filename_rad,
                       prefix='cutted')
        
