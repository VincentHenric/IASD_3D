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
import time
import os

import subsampling

def local_PCA(points):
    
    bary = points.mean(axis=0, keepdims=True)
    
    Q = points - bary
    
    cov = 1/len(points) * (Q.T @ Q)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


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

def get_general_features(query_points_indices, cloud_points, colors, intensities, radius):
    n = len(query_points_indices)
    epsilon = 1e-16
    
    # prepare arrays
    all_eigenvalues = np.zeros((n, 3))
    all_normals = np.zeros((n, 3))
    all_geometric= np.zeros((n, 3))
    all_intensities = np.zeros((n, 1))
    
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kdtree = KDTree(cloud_points)  
    
    neighborhood_indices = kdtree.query_radius(cloud_points[query_points_indices], r=radius, count_only=False, return_distance=False)
    
    for i, neighbor_index_list in enumerate(neighborhood_indices):
        neighborhoods = cloud_points[neighbor_index_list,:]
        eigenvalues, eigenvectors = local_PCA(neighborhoods)
        all_eigenvalues[i,:] = eigenvalues
        all_normals[i,:] = eigenvectors[:,0]
        all_intensities[i] = compute_entropy(intensities[neighbor_index_list], bins=256, range=(0,256))
        
    all_geometric[:,0] = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2]+epsilon)
    all_geometric[:,1] = (all_eigenvalues[:,1]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+epsilon)
    all_geometric[:,2] = all_eigenvalues[:,0]/(all_eigenvalues[:,2]+epsilon)
    
    return all_normals, all_geometric, all_intensities, neighborhood_indices

def compute_features(query_points_indices, cloud_points, colors, intensities, radius):
    epsilon = 1e-16

    # Initialize covariance matrices
    query_cov = np.zeros((len(query_points_indices), 10, 10))
    
    # preprocessing for colors
    if colors.dtype in [np.uint8, 'uint8'] or colors.max()>1:
        colors = colors/255
    
    # Compute the features for all query points in the cloud
    all_normals, all_geometric, all_intensities, neighborhood_indices = get_general_features(np.arange(len(cloud_points)), cloud_points, colors, intensities, radius)
    
    # take indices for query points
    neighborhood_indices = neighborhood_indices[query_points_indices]
    
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
        
        # get remaining features
        alphas = pi_p_vectors @ all_normals[query_point_index].T
        betas = (pi_p_vectors * all_normals[neighbor_index_list]).sum(axis=1)
        gammas = all_normals[neighbor_index_list] @ all_normals[query_point_index].T
        ent_intensities = all_intensities[neighbor_index_list].copy()
        
        # normalization
        alphas = (alphas-alphas.min())/(alphas.max()-alphas.min()+epsilon)
        betas = (betas-betas.min())/(betas.max()-betas.min()+epsilon)
        gammas = (gammas-gammas.min())/(gammas.max()-gammas.min()+epsilon)
        ent_intensities = (ent_intensities-ent_intensities.min())/(ent_intensities.max()-ent_intensities.min()+epsilon)

        # concatenation
        features = np.concatenate((all_geometric[neighbor_index_list],
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




#def compute_features_2(query_points_indices, cloud_points, colors, intensities, radius):
#    epsilon = 1e-16
#
#    # Initialize covariance matrices
#    query_cov = np.zeros((len(query_points_indices), 10, 10))
#    
#    # preprocessing for colors
#    if colors.dtype in [np.uint8, 'uint8'] or colors.max()>1:
#        colors = colors/255
#    
#    
#    kdtree = KDTree(cloud_points)  
#    
#    # take indices for query points
#    neighborhood_indices = kdtree.query_radius(cloud_points[query_points_indices], r=radius, count_only=False, return_distance=False)
#    all_neighborhood_indices, all_neighborhood_local_indices = np.unique(np.concatenate(neighborhood_indices), return_index=True)
#
#    
#    # compute remaining features and 
#    for i, neighbor_index_list in enumerate(neighborhood_indices):
#        query_point_index = query_points_indices[i]
#        all_normals, all_geometric, all_intensities, _ = get_general_features(all_neighborhood_indices, cloud_points, colors, intensities, radius)
#        
#        pi_p_vectors = cloud_points[neighbor_index_list]-cloud_points[query_point_index]
#        pi_p_norm = np.linalg.norm(pi_p_vectors, axis=1, keepdims=True)
#        indices_to_remove = np.where(pi_p_norm==0)[0]
#        neighbor_index_list = np.delete(neighbor_index_list, indices_to_remove)
#        
#        pi_p_vectors = pi_p_vectors[pi_p_norm[:,0]!=0,:] / pi_p_norm[pi_p_norm[:,0]!=0]
#        
#        if len(neighbor_index_list)<2:
#            continue
#        
#        # get remaining features
#        alphas = pi_p_vectors @ all_normals[query_point_index].T
#        betas = (pi_p_vectors * all_normals[neighbor_index_list]).sum(axis=1)
#        gammas = all_normals[neighbor_index_list] @ all_normals[query_point_index].T
#        intensities = all_intensities[neighbor_index_list]
#        
#        # normalization
#        alphas = (alphas-alphas.min())/(alphas.max()-alphas.min()+epsilon)
#        betas = (betas-betas.min())/(betas.max()-betas.min()+epsilon)
#        gammas = (gammas-gammas.min())/(gammas.max()-gammas.min()+epsilon)
#        intensities = (intensities-intensities.min())/(intensities.max()-intensities.min()+epsilon)
#
#        # concatenation
#        features = np.concatenate((all_geometric[neighbor_index_list],
#                                   alphas[:,None],
#                                   betas[:,None],
#                                   gammas[:,None],
#                                   colors[neighbor_index_list],
#                                   intensities), axis=1)
#        
#        bary = features.mean(axis=0, keepdims=True)
#        Q = features - bary
#        cov = 1/(len(features)-1) * (Q.T @ Q)
#        
#        query_cov[i,:] = cov
#
#    return query_cov

def add_noise(cloud_points, sigma=0.001):
    noise = np.random.normal(size=(len(cloud_points),3), scale=sigma)
    return cloud_points + noise

def select_top_indices(query_cov, q=0.95):
    dets = np.abs(np.linalg.det(query_cov))
    quantile = np.quantile(dets, q)
    return np.where(dets>quantile)[0]

def compute_r_bar(cloud_points):
    kdtree = KDTree(cloud_points)  
    dist, ind = kdtree.query(cloud_points, k=2, return_distance=True)

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
        name = filename.split('.')[0]
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
                                     radius)
        dets = np.linalg.det(query_cov)
        
        # keep best points
        thresh = np.quantile(dets, quantile)
        sampled_query_points_indices = np.where(dets>thresh)[0]
        query_points_indices = sampling_indices[sampled_query_points_indices]
        
        # Save best points
        np.save(os.path.join(saved_data_path, 'covariance_{}.npy'.format(name)), query_cov[sampled_query_points_indices])
        np.save(os.path.join(saved_data_path, 'query_indices_{}.npy'.format(name)), query_points_indices)
        
        #write_ply('../data/Lille_street_small_featured_{}.ply'.format(radius), [new_cloud], ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])



    # Features computation
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        filename = 'bildstein_station1_xyz_intensity_rgb_cutted.ply'
        filename = 'bunny_original.ply'
        name = filename.split('.')[0]
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.zeros((len(cloud_points),3))
        intensities = np.ones((len(cloud_points),1))
        
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        
        
        # Load points of interest
        query_points_indices = np.load(os.path.join(data_path, 'query_indices_{}.npy'.format(name)))
        query_points_indices = np.arange(len(cloud_points))
        
        # Parameters
        radius = 0.3
        radius = 0.02
        noise = 0.001
        query_points_indices = np.array([1,10,100])
        
        # Preprocessing
        cloud_points = add_noise(cloud_points, noise)
        
        # Computations
        query_cov = compute_features(query_points_indices, cloud_points, colors, intensities, radius)
        
        # Save results
        np.save(os.path.join(saved_data_path, 'covariance_{}_radius-{}_noise-{}.npy'.format(name, radius, noise)), query_cov)
        
        #write_ply('../data/Lille_street_small_featured_{}.ply'.format(radius), [new_cloud], ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])


    # Features analysis
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        filename = 'bildstein_station3_xyz_intensity_rgb_cutted.ply'
        radius=0.3
        noise=0.001
        nb_values = 10
        noise_2 = 2
        name = filename.split('.')[0]
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        #covariances = np.load(os.path.join(saved_data_path, 'covariance_{}_radius-{}_noise-{}.npy'.format(name, radius, noise)))
        covariances = np.load(os.path.join(saved_data_path, 'covariance_{}_radius-{}_noise-{}-{}-{}.npy'.format(name, radius, noise, nb_values, noise_2)))

        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.vstack((cloud_ply['red'], cloud_ply['green'], cloud_ply['blue'])).T
        intensities = cloud_ply['intensity']
        
        
        # analysis
        #determinants = np.linalg.det(covariances)
        
        #plt.hist(determinants[:100], bins=50)
        
        #indices = select_top_indices(query_cov[:,:6,:6], q=0.95)
        indices = select_top_indices(covariances, q=0.95)
        
        sailant = np.zeros((len(cloud_points),1))
        sailant[indices] = 1
  
#        write_ply(os.path.join(saved_data_path, 'sailant_{}_radius-{}_noise-{}.ply'.format(name, radius, noise)),
#                  (cloud_points, intensities[:,None], colors, sailant),
#                  ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'sailant'])
        
        write_ply(os.path.join(saved_data_path, 'sailant_{}_radius-{}_noise-{}-{}-{}.ply'.format(name, radius, noise, nb_values, noise_2)),
                  (cloud_points, intensities[:,None], colors, sailant),
                  ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'sailant'])

        
        