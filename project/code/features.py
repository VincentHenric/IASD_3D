#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:40:41 2020

@author: henric
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from scipy.stats import entropy
from utils.ply import write_ply, read_ply
from fileutils import dict_to_str, give_filename, parse_filename
import subsampling

import time
import os

def local_PCA(points):
    
    bary = points.mean(axis=0, keepdims=True)
    
    Q = points - bary
    
    cov = 1/len(points) * (Q.T @ Q)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return np.abs(eigenvalues), eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius):

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kdtree = KDTree(cloud_points)  
    
    neighborhood_indices = kdtree.query_radius(query_points, r=radius, count_only=False, return_distance=False)
    
    for i, neighbor_index_list in enumerate(neighborhood_indices):
        neighborhoods = cloud_points[neighbor_index_list,:]
        eigenvalues, eigenvectors = local_PCA(neighborhoods)
        all_eigenvalues[i,:] = eigenvalues
        all_eigenvectors[i,:] = eigenvectors
    
    return all_eigenvalues, all_eigenvectors

def get_general_features(query_points_indices, cloud_points, colors, intensities, kdtree, query_type='radius', radius=None, nb_neighbors=None):
    n = len(query_points_indices)
    epsilon = 1e-16
    
    # prepare arrays
    all_eigenvalues = np.zeros((n, 3))
    all_normals = np.zeros((n, 3))
    all_geometric= np.zeros((n, 3))
    all_intensities = np.zeros((n, 1))  
    
    neighborhood_indices = query_kdtree(cloud_points[query_points_indices], kdtree, query_type=query_type, radius=radius, nb_neighbors=nb_neighbors)
    
    for i, neighbor_index_list in enumerate(neighborhood_indices):
        neighborhoods = cloud_points[neighbor_index_list,:]
        eigenvalues, eigenvectors = local_PCA(neighborhoods)
        all_eigenvalues[i,:] = eigenvalues
        all_normals[i,:] = eigenvectors[:,0]
        all_intensities[i] = compute_entropy(intensities[neighbor_index_list], bins=256, range=(0,256))
        
    all_geometric[:,0] = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2]+epsilon)
    all_geometric[:,1] = (all_eigenvalues[:,1]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+epsilon)
    all_geometric[:,2] = all_eigenvalues[:,0]/(all_eigenvalues[:,2]+epsilon)
    
    return all_normals, all_geometric, all_intensities

def query_kdtree(cloud_points, kdtree, query_type='radius', radius=None, nb_neighbors=None):
    if query_type == 'radius':
        return kdtree.query_radius(cloud_points, r=radius, count_only=False, return_distance=False)
    elif query_type == 'neighbors':
        return kdtree.query(cloud_points, k=nb_neighbors, return_distance=False)
    else:
        raise ValueError('wrong query type')

def compute_features(query_points_indices, cloud_points, colors, intensities, query_type='radius', radius=None, nb_neighbors=None):
    epsilon = 1e-16

    # Initialize covariance matrices
    query_cov = np.zeros((len(query_points_indices), 10, 10))
    
    # preprocessing for colors
    if colors.dtype in [np.uint8, 'uint8'] or colors.max()>1:
        colors = colors/255
        
    # We create the kdtree
    kdtree = KDTree(cloud_points)
    neighborhood_indices = query_kdtree(cloud_points[query_points_indices], kdtree, query_type=query_type, radius=radius, nb_neighbors=nb_neighbors)
    all_neighborhood_indices = sorted(set(flatten(neighborhood_indices)))
    decode = {k:v for v,k in enumerate(all_neighborhood_indices)}
    
    # Compute the features for all query points in the cloud
    all_normals, all_geometric, all_intensities = get_general_features(all_neighborhood_indices, cloud_points, colors, intensities, kdtree, query_type=query_type, radius=radius, nb_neighbors=nb_neighbors)
    
    # take indices for query points
    #neighborhood_indices = neighborhood_indices[query_points_indices]
    
    # compute remaining features and 
    for i, neighbor_index_list in enumerate(neighborhood_indices):
        query_point_index = query_points_indices[i]
        
        pi_p_vectors = cloud_points[neighbor_index_list]-cloud_points[query_point_index]
        pi_p_norm = np.linalg.norm(pi_p_vectors, axis=1, keepdims=True)
        indices_to_remove = np.where(pi_p_norm==0)[0]
        neighbor_index_list = np.delete(neighbor_index_list, indices_to_remove)
        
        pi_p_vectors = pi_p_vectors[pi_p_norm[:,0]!=0,:] / pi_p_norm[pi_p_norm[:,0]!=0]
        
        if len(neighbor_index_list)<2:
            continue
        
        # take corresponding indices in sublist
        neighbor_index_list_2 = [decode[neighbor_index] for neighbor_index in neighbor_index_list]
        query_point_index_2 = decode[query_points_indices[i]]
        
        # get remaining features
        alphas = pi_p_vectors @ all_normals[query_point_index_2].T
        betas = (pi_p_vectors * all_normals[neighbor_index_list_2]).sum(axis=1)
        gammas = all_normals[neighbor_index_list_2] @ all_normals[query_point_index_2].T
        ent_intensities = all_intensities[neighbor_index_list_2].copy()
        
        alphas = np.arccos(np.clip(np.abs(alphas), 0, 1))
        betas = np.arccos(np.clip(np.abs(betas), 0, 1))
        gammas = np.arccos(np.clip(np.abs(gammas), 0, 1))
        
        # normalization
        alphas = (alphas-alphas.min())/(alphas.max()-alphas.min()+epsilon)
        betas = (betas-betas.min())/(betas.max()-betas.min()+epsilon)
        gammas = (gammas-gammas.min())/(gammas.max()-gammas.min()+epsilon)
        ent_intensities = (ent_intensities-ent_intensities.min())/(ent_intensities.max()-ent_intensities.min()+epsilon)

        # concatenation
        features = np.concatenate((all_geometric[neighbor_index_list_2],
                                   alphas[:,None],
                                   betas[:,None],
                                   gammas[:,None],
                                   colors[neighbor_index_list],
                                   ent_intensities), axis=1)
        
        bary = features.mean(axis=0, keepdims=True)
        Q = features - bary
        cov = 1/(len(features)-1) * (Q.T @ Q)
        
        query_cov[i,:] = cov

    return query_cov

def flatten(l):
    return [item for sublist in l for item in sublist]

def select_top_indices(query_cov, q=0.95):
    dets = np.abs(np.linalg.det(query_cov))
    quantile = np.quantile(dets, q)
    return np.where(dets>quantile)[0]

def compute_r_bar(cloud_points):
    kdtree = KDTree(cloud_points)  
    dist, ind = kdtree.query(cloud_points, k=2, return_distance=True)
    return dist[:,1].mean()

def compute_entropy(intensities, **kwargs):
    hist = np.histogram(intensities, density=True, **kwargs)[0]
    return entropy(hist)

def identify_good_radius(cloud_points, radiuses=[0.1, 0.3, 0.5, 1, 5], N=100000, **kwargs):
    kdtree = KDTree(cloud_points)  

    indices = np.random.randint(len(cloud_points), size=N)  
    
    sizes = {}
    for radius in radiuses:
        size = kdtree.query_radius(cloud_points[indices,:], r=radius, count_only=True, return_distance=False)
        lim = np.quantile(size, 0.95)
        low_lim = np.quantile(size, 0.05)
        plt.clf
        plt.hist(size, range=(0,lim), **kwargs)
        plt.axvline(low_lim, color='red')
        plt.text(low_lim,-3,low_lim, color='red')
        #plt.hist(size, range=(0,200))
        plt.show()
        sizes[radius] = size
    return sizes

def normalize_intensities(intensities, low=-2048, high=2047):
    intensities = (intensities - low)/(high-low)*255
    return intensities

def ACOV(cloud_points, colors, intensities, radius, t0=0.268, b=1.3, q=0.95, factor=0.2):
    params = {'subs':'decimation', 'factor':factor}
    sub_points, sub_colors, sub_intensities, sub_indices = subsampling.subsample(cloud_points, colors, intensities, **params)
    
    subcovariances = compute_features(np.arange(len(sub_points)), sub_points, sub_colors, sub_intensities, query_type='radius', radius=radius)
    keypoints = select_top_indices(subcovariances , q=q)
    key_point_indices = np.array(sub_indices)[keypoints]
    
    n = len(key_point_indices)
    m_values = list(range(6))
    current_dets = np.zeros(n)
    current_covariances = np.zeros((n, 10, 10))
    current_m = m_values[0] * np.ones(n)
    
    for m in m_values:
        radius = t0 * b**m
        covariances = compute_features(key_point_indices, cloud_points, colors, intensities, query_type='radius', radius=radius)
        dets = np.abs(np.linalg.det(covariances))
        change = (dets > current_dets)
        current_covariances = change[:,None,None]*covariances + (~change[:,None,None])*current_covariances
        current_m = change*m + (~change)*current_m
        current_dets = np.maximum(current_dets, dets)
    
    return current_covariances, current_m, key_point_indices
        

# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':
    
    data_path = '../data'
    saved_data_path = '../saved_data'
    r_bar = 0.0134
    r_bar = 0.0224
                
    if False:
        
        # Load cloud as a [N x 3] matrix
        filename = 'bildstein1_factor:0.2-noise:0.001-subs:decimation.ply'
        #filename = 'bunny-original.ply'
        
        prefix, name, params = parse_filename(filename)
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.zeros((len(cloud_points),3))
        intensities = np.ones((len(cloud_points),1))
        
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        
        
        # Load points of interest
        query_points_indices = np.arange(len(cloud_points))
        
        # Parameters
        params.update({'radius':0.3})
        
        # Computations        
        all_normals, all_geometric, all_intensities, _ = get_general_features(query_points_indices, cloud_points, colors, intensities, params['radius'])

        filename_features = give_filename(name, prefix='features', params=params, extension='ply')
        write_ply(os.path.join(saved_data_path, filename_features),
                  (cloud_points, colors, all_normals, all_geometric, all_intensities),
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz', 'linearity', 'planarity', 'sphericity', 'intensity_entropy']
                  )
        
    
    if False:
        
        # Load cloud as a [N x 3] matrix
        filename = 'features_bildstein3_factor:0.2-noise:0.001-radius:0.3-subs:decimation.ply'
        #filename = 'bunny-original.ply'
        
        prefix, name, params = parse_filename(filename)
        cloud_path = os.path.join(saved_data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        all_intensities = cloud_ply['intensity_entropy']
        all_normals = np.vstack((cloud_ply['nx'], cloud_ply['ny'], cloud_ply['nz'])).T
        all_geometric = np.vstack((cloud_ply['linearity'], cloud_ply['planarity'], cloud_ply['sphericity'])).T
        
        
        # Load points of interest
        query_points_indices = np.arange(len(cloud_points))
        
        # Parameters
        params.update({'radius':0.3})
        
        # Computations        
        all_normals, all_geometric, all_intensities, _ = get_general_features(query_points_indices, cloud_points, colors, intensities, params['radius'])

        filename_features = give_filename(name, prefix='features', params=params, extension='ply')
        write_ply(os.path.join(saved_data_path, filename_features),
                  (cloud_points, colors, all_normals, all_geometric, all_intensities),
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz', 'linearity', 'planarity', 'sphericity', 'intensity_entropy']
                  )


    # Find a good radius for neighbors
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        filename = 'bunny_original.ply'
        #filename = 'Lille_street_small.ply'
        #filename = 'bildstein_station1_xyz_intensity_rgb_cutted.ply'
        name = filename.split('.')[0]
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T


        # Parameters
        radiuses = [0.1, 0.3, 0.5, 1, 5]
        
        radiuses = [0.01, 0.02, 0.03]

        # plot the number of neighbors
        sizes = identify_good_radius(cloud_points, radiuses=radiuses, N=100000, bins=100)
        

    # Uniform sampling to find interesting points
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        filename = 'bunny_original.ply'
        #filename = 'Lille_street_small.ply'
        #filename = 'bildstein_station1_xyz_intensity_rgb_cutted.ply'
        filename = 'cutted_bildstein1.ply'
        prefix, name, params = parse_filename(filename)
        
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        #colors = np.zeros((len(cloud_points),3))
        #intensities = np.ones((len(cloud_points),1))
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        

        # Parameters
        radius = 0.02
        quantile = 0.9
        percent = 0.1
        sampling_indices = np.random.randint(len(cloud_points), size=round(percent*len(cloud_points)))
        
        all_normals, all_geometric, all_intensities, neighborhood_indices = get_general_features(cloud_points, colors, intensities, radius)

        
        # Computations on all points of the sampling
        query_cov = compute_features(np.array(range(len(sampling_indices))),
                                     cloud_points[sampling_indices],
                                     colors[sampling_indices],
                                     intensities[sampling_indices],
                                     query_type='radius',
                                     radius=radius)
        dets = np.linalg.det(query_cov)
        
        # keep best points
        thresh = np.quantile(dets, quantile)
        sampled_query_points_indices = np.where(dets>thresh)[0]
        query_points_indices = sampling_indices[sampled_query_points_indices]
        
        # Save best points
        cov_filename = give_filename(name, prefix='covariance', params=params, extension='npy')
        indices_filename = give_filename(name, prefix='query-indices', params=params, extension='npy')
        
        np.save(os.path.join(saved_data_path, cov_filename), query_cov[sampled_query_points_indices])
        np.save(os.path.join(saved_data_path, indices_filename), query_points_indices)
        
        #write_ply('../data/Lille_street_small_featured_{}.ply'.format(radius), [new_cloud], ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])



    # Features computation
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        filename = 'bildstein3_factor:0.2-noise:0.001-subs:decimation.ply'
        #filename = 'bunny-original.ply'
        
        prefix, name, params = parse_filename(filename)
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.zeros((len(cloud_points),3))
        intensities = np.ones((len(cloud_points),1))
        
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        
        
        # Load points of interest
        #indices_filename = give_filename(name, prefix='query-indices', params=params, extension='npy')
        #query_points_indices = np.load(os.path.join(data_path, indices_filename))
        query_points_indices = np.arange(len(cloud_points))
        
        # Parameters
        params.update({'radius':0.3})
#        radius = 0.3
#        radius = 0.02
#        noise = 0.001
        #query_points_indices = np.array([1,10,100])
        
        # Computations
        query_cov = compute_features(query_points_indices, cloud_points, colors, intensities, query_type='radius', radius=params['radius'])
        
        # Save results
        cov_filename = give_filename(name, prefix='covariance', params=params, extension='npy')
        np.save(os.path.join(saved_data_path, cov_filename), query_cov)
        
        #write_ply('../data/Lille_street_small_featured_{}.ply'.format(radius), [new_cloud], ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])

    # Features computation with ACOV approach
    # ********************
    #

    if True:
        filename = 'cutted_bildstein1.ply'
        prefix, name, params = parse_filename(filename)
        
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        # parameters
        #params = {'subs':'grid', 'voxel_size':0.2}
        params = {'subs':'decimation', 'factor':0.5, 'radius':0.3, 'q':0.95}
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        
        current_covariances, current_m, key_point_indices = ACOV(cloud_points, colors, intensities, radius=params['radius'], t0=0.268, b=1.3, q=params['q'], factor=params['factor'])

        cov_filename = give_filename(name, prefix='covariance', params=params, extension='npy')
        np.save(os.path.join(saved_data_path, cov_filename), current_covariances)
        
        #radius_filename = give_filename(name, prefix='radius', params=params, extension='npy')
        #np.save(os.path.join(saved_data_path, radius_filename), query_cov)
        
        keypoints = np.zeros((len(cloud_points),1))
        keypoints[key_point_indices] = 1
        
        keypoints_filename = give_filename(name, prefix='keypoints', params=params, extension='ply')
        write_ply(os.path.join(saved_data_path, keypoints_filename),
                  (cloud_points, intensities[:,None], colors, keypoints),
                  ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'keypoints'])

    # Features analysis
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        filename = 'bildstein1_factor:0.2-noise:0.001-subs:decimation.ply'
        cov_params = {'radius':0.3}
        prefix, name, params = parse_filename(filename)
        
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cov_params.update(params)
        cov_filename = give_filename(name, prefix='covariance', params=cov_params, extension='npy')
        covariances = np.load(os.path.join(saved_data_path, cov_filename))

        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        
        
        # analysis
        #determinants = np.linalg.det(covariances)
        
        #plt.hist(determinants[:100], bins=50)
        
        #indices = select_top_indices(query_cov[:,:6,:6], q=0.95)
        quantile = 0.8
        cov_params.update({'q':quantile})
        indices = select_top_indices(covariances, q=cov_params['q'])
        
        keypoints = np.zeros((len(cloud_points),1))
        keypoints[indices] = 1
  
#        write_ply(os.path.join(saved_data_path, 'sailant_{}_radius-{}_noise-{}.ply'.format(name, radius, noise)),
#                  (cloud_points, intensities[:,None], colors, sailant),
#                  ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'sailant'])
        
        keypoints_filename = give_filename(name, prefix='keypoints', params=cov_params, extension='ply')
        write_ply(os.path.join(saved_data_path, keypoints_filename),
                  (cloud_points, intensities[:,None], colors, keypoints),
                  ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'keypoints'])

        
        