#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:12:23 2020

@author: henric
"""
import numpy as np
from collections import defaultdict
import os

from utils.ply import write_ply, read_ply, check_with_colors_and_intensity
from fileutils import dict_to_str, give_filename, parse_filename
import features

def grid_subsample(cloud_points, colors, intensities, voxel_size, **kwargs):
    voxel_indices = np.floor(cloud_points/voxel_size).astype('int')
    
    stats = defaultdict(lambda: ((np.zeros(3),np.zeros(3),np.zeros(3),0)))

    for i, voxel_index in enumerate(voxel_indices):
        sum_p, sum_col, sum_int, c = stats[tuple(voxel_index)]
        stats[tuple(voxel_index)] = (sum_p+cloud_points[i], sum_col+colors[i], sum_int+intensities[i], c+1)

    subsampled_points = np.array([v[0]/v[-1] for v in stats.values()])
    subsampled_colors = np.array([np.clip((v[1]/v[-1]).astype('uint8'),0,255) for v in stats.values()])
    subsampled_intensities = np.array([np.clip((v[2]/v[-1]).astype('uint8'),0,255) for v in stats.values()])
    
    return subsampled_points, subsampled_colors, subsampled_intensities, None

def cloud_decimation(cloud_points, colors, intensities, factor, **kwargs):

    # YOUR CODE
    factor = int(1/factor)
    decimated_points = cloud_points[0:-1:factor,:]
    decimated_colors = colors[0:-1:factor,:]
    decimated_intensities = intensities[0:-1:factor]

    return decimated_points, decimated_colors, decimated_intensities, list(range(0,len(cloud_points),factor))

def subsample(cloud_points, colors, intensities, **kwargs):
    subs = kwargs.get('subs','grid')
    if subs=='grid':
        return grid_subsample(cloud_points, colors, intensities, **kwargs)
    elif subs=='decimation':
        return cloud_decimation(cloud_points, colors, intensities, **kwargs)
    else:
        raise ValueError('method not recognized')

def add_noise(cloud_points, sigma=0.001):
    """
    add noise to cloud points
    """
    noise = np.random.normal(size=(len(cloud_points),3), scale=sigma)
    return cloud_points + noise


if __name__ == '__main__':  
    data_path = '../data'
    saved_data_path = '../saved_data'
    
    # Get a subsampled and noisy dataset
    # ********************
    #
    
    if True:
    
        # Load cloud as a [N x 3] matrix
        filename = 'cutted_bildstein1.ply'
        prefix, name, params = parse_filename(filename)

        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.zeros((len(cloud_points),3))
        intensities = np.ones((len(cloud_points),1))
        
        with_colors = check_with_colors_and_intensity(cloud_ply)
        if with_colors:
            colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
            intensities = cloud_ply['intensity']
        
        # parameters
        params = {}
        #params = {'subs':'grid', 'voxel_size':0.2}
        params = {'subs':'decimation', 'factor':0.2}
        r_bar = features.compute_r_bar(cloud_points)
        
        cloud_points, colors, intensities, _ = subsample(cloud_points, colors, intensities, **params)
        
        # Preprocessing
        params.update({'noise':np.round(2*r_bar,3)})
        cloud_points = add_noise(cloud_points, params['noise'])
        
        # Save results
        filename = give_filename(name+'-'+dict_to_str(params), '', {})
        if with_colors:
            write_ply(os.path.join(data_path, filename),
                      (cloud_points, intensities[:,None], colors),
                      ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue'])
        else:
            write_ply(os.path.join(data_path, filename),
                      (cloud_points),
                      ['x', 'y', 'z'])
