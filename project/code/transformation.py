#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:18:21 2020

@author: henric
"""


import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree

from utils.ply import write_ply, read_ply, check_with_colors_and_intensity
from fileutils import dict_to_str, give_filename, parse_filename

import os

def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    bary = data.mean(axis=1, keepdims=True)
    bary_ref = ref.mean(axis=1, keepdims=True)
    
    Q = data - bary
    Q_ref = ref - bary_ref
    
    H = Q @ Q_ref.T
    
    U, _, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    if np.linalg.det(R)<0:
        U[-1] *= -1 
        R = Vt.T @ U.T
    T = bary_ref - R @ bary

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = []
    
    # YOUR CODE
    i = 0
    old_R = np.identity(len(data))
    old_T = np.zeros((len(data), 1))
    
    rms = RMS_threshold + 1
    old_rms = None
    
    kdtree = KDTree(ref.T)
    
    while i<max_iter and rms > RMS_threshold:
        
        neighborhood_indices = kdtree.query(data_aligned.T, k=1, return_distance=False).squeeze()
        neighborhoods = ref[:,neighborhood_indices]

        R, T = best_rigid_transform(data_aligned, neighborhoods)
        
        data_aligned = R @ data_aligned + T

        rms = np.sqrt(np.mean(np.linalg.norm(neighborhoods-data_aligned, axis=0)))
        
        R_list.append(old_R @ R)
        T_list.append(R @ old_T + T)
        neighbors_list.append(neighborhood_indices)
        rms_list.append(rms)
        
        if old_rms:
            if old_rms-rms<RMS_threshold:
                break
        
        old_rms = rms
        old_R = old_R @ R
        old_T = R @ old_T + T
        
        i += 1

    return data_aligned, R_list, T_list, neighbors_list, rms_list

def icp_point_to_point_stochastic(data, ref, max_iter, RMS_threshold, sampling_limit=1000, final_overlap=1):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = []
    
    # YOUR CODE
    i = 0
    old_R = np.identity(len(data))
    old_T = np.zeros((len(data), 1))
    
    rms = RMS_threshold + 1
    kdtree = KDTree(ref.T)
    
    while i<max_iter and rms > RMS_threshold:
        
        data_aligned_sampled_indices = np.random.choice(np.arange(data_aligned.shape[1]), size=sampling_limit, replace=False)
        data_aligned_sampled = data_aligned[:, data_aligned_sampled_indices]
        
        distances, neighborhood_indices = kdtree.query(data_aligned_sampled.T, k=1, return_distance=True)
        distances = distances.squeeze()
        neighborhood_indices = neighborhood_indices.squeeze()
        neighborhoods = ref[:,neighborhood_indices]
        
        if final_overlap != 1:
            nb_indices_to_select = int(len(distances) * final_overlap)
            kept_indices = np.argpartition(distances, nb_indices_to_select)[:nb_indices_to_select]
            data_aligned_sampled2 = data_aligned_sampled[:, kept_indices]
            neighborhoods2 = neighborhoods[:, kept_indices]
        else:
            data_aligned_sampled2 = data_aligned_sampled
            neighborhoods2 = neighborhoods
        
        R, T = best_rigid_transform(data_aligned_sampled2, neighborhoods2)
        
        data_aligned = R @ data_aligned + T

        rms = np.sqrt(np.mean(np.linalg.norm(neighborhoods-data_aligned_sampled, axis=0)))
        
        R_list.append(old_R @ R)
        T_list.append(R @ old_T + T)
        neighbors_list.append(neighborhood_indices)
        rms_list.append(rms)
        
        old_R = old_R @ R
        old_T = R @ old_T + T
        
        i += 1

    return data_aligned, R_list, T_list, neighbors_list, rms_list



if __name__ == '__main__':
    data_path = '../data'
    saved_data_path = '../saved_data'
    
    if True:
        filename1 = 'keypoints_bildstein1_factor:0.2-geom:0-q:0.98-radius:0.561-subs:decimation-t0:0.449.ply'
        prefix1, name1, params1 = parse_filename(filename1)
        cloud_ply1 = read_ply(os.path.join(saved_data_path, filename1))
        cloud_points1 = np.vstack((cloud_ply1['x'], cloud_ply1['y'], cloud_ply1['z'])).T
        colors1 = np.zeros((len(cloud_points1),3))
        intensities1 = np.ones((len(cloud_points1),1))
        with_colors1 = check_with_colors_and_intensity(cloud_ply1)
        if with_colors1:
            colors1 = np.vstack((cloud_ply1['red'], cloud_ply1['green'], cloud_ply1['blue'])).T
            intensities1 = cloud_ply1['intensity']
        keypoints1 = cloud_ply1['keypoints']
        
        filename2 = 'keypoints_bildstein3_factor:0.2-geom:0-q:0.98-radius:0.634-subs:decimation-t0:0.507.ply'
        prefix2, name2, params2 = parse_filename(filename2)
        cloud_ply2 = read_ply(os.path.join(saved_data_path, filename2))
        cloud_points2 = np.vstack((cloud_ply2['x'], cloud_ply2['y'], cloud_ply2['z'])).T
        colors2 = np.zeros((len(cloud_points2),3))
        intensities2 = np.ones((len(cloud_points2),1))
        with_colors2 = check_with_colors_and_intensity(cloud_ply2)
        if with_colors2:
            colors2 = np.vstack((cloud_ply2['red'], cloud_ply2['green'], cloud_ply2['blue'])).T
            intensities2 = cloud_ply2['intensity']
        keypoints2 = cloud_ply2['keypoints']
        
        params = {k+'1':v for k,v in params1.items()}
        params.update({k+'2':v for k,v in params2.items()})
        correspondences_filename = give_filename(name1+'-'+name2, prefix='corresp-ratio:1.3', params=params, extension='npy')
        correspondences_indices = np.load(os.path.join(saved_data_path, correspondences_filename)).astype('int')
        
        matched_points1 = np.zeros((len(cloud_points1),1))
        matched_points1[correspondences_indices[:,0]] = 1
        matched_points2 = np.zeros((len(cloud_points2),1))
        matched_points2[correspondences_indices[:,1]] = 1
        
        max_iter = 10000
        RMS_threshold = 1e-5
        
        ref = cloud_points1[correspondences_indices[:,0]].T
        data = cloud_points2[correspondences_indices[:,1]].T
        
        R, T = best_rigid_transform(data, ref)
        
        new_cloud_points2 = (R @ cloud_points2.T + T).T
        
        # Save cloud
        registered_filename = give_filename(name1+'-'+name2, prefix='registered2', params=params, extension='ply')
        if with_colors2:
            write_ply(os.path.join(saved_data_path, registered_filename),
                      (new_cloud_points2, intensities2, colors2, keypoints2, matched_points2),
                      ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'keypoints', 'matchedpoints'])
        else:
            write_ply(os.path.join(saved_data_path, registered_filename),
                      (new_cloud_points2, keypoints2, matched_points2),
                      ['x', 'y', 'z', 'keypoints', 'matchedpoints'])
        
        registered_filename = give_filename(name1+'-'+name2, prefix='registered1', params=params, extension='ply')
        if with_colors1:
            write_ply(os.path.join(saved_data_path, registered_filename),
                  (cloud_points1, intensities1, colors1, keypoints1, matched_points1),
                  ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'keypoints', 'matchedpoints'])
        else:
            write_ply(os.path.join(saved_data_path, registered_filename),
                  (cloud_points1, keypoints1, matched_points1),
                  ['x', 'y', 'z', 'keypoints', 'matchedpoints'])
        

        
        
        # Compute RMS
        rms = np.sqrt(np.mean(np.linalg.norm(ref-new_cloud_points2[correspondences_indices[:,1]].T, axis=0)))
        # Print RMS
        print('RMS = {}'.format(rms))