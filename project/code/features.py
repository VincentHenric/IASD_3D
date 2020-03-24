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

def get_general_features(cloud_points, colors, intensities, radius):
    n = len(cloud_points)
    epsilon = 1e-16
    
    # prepare arrays
    all_eigenvalues = np.zeros((n, 3))
    all_normals = np.zeros((n, 3))
    all_geometric= np.zeros((n, 3))
    all_intensities = np.zeros((n, 1))
    
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    kdtree = KDTree(cloud_points)  
    
    neighborhood_indices = kdtree.query_radius(cloud_points, r=radius, count_only=False, return_distance=False)
    
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
    if colors.dtype in [np.uint8, 'uint8']:
        colors = colors/255
    
    # Compute the features for all query points in the cloud
    all_normals, all_geometric, all_intensities, neighborhood_indices = get_general_features(cloud_points, colors, intensities, radius)
    
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
        intensities = all_intensities[neighbor_index_list]
        
        # normalization
        alphas = (alphas-alphas.min())/(alphas.max()-alphas.min()+epsilon)
        betas = (betas-betas.min())/(betas.max()-betas.min()+epsilon)
        gammas = (gammas-gammas.min())/(gammas.max()-gammas.min()+epsilon)
        intensities = (alphas-alphas.min())/(alphas.max()-alphas.min()+epsilon)

        # concatenation
        features = np.concatenate((all_geometric[neighbor_index_list],
                                   alphas[:,None],
                                   betas[:,None],
                                   gammas[:,None],
                                   colors[neighbor_index_list],
                                   intensities[:,None]), axis=1)
        
        bary = features.mean(axis=0, keepdims=True)
        Q = features - bary
        cov = 1/(len(features)-1) * (Q.T @ Q)
        
        query_cov[i,:] = cov

    return query_cov

def compute_entropy(intensities, **kwargs):
    hist = np.histogram(intensities, density=True, **kwargs)[0]
    return entropy(hist)


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

    # Uniform sampling to find interesting points
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        filename = 'bunny_original.ply'
        filename = 'Lille_street_small.ply'
        name = filename.split('.')[0]
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.zeros((len(cloud_points),3))
        intensities = np.ones((len(cloud_points),1))
        
        colors = np.vstack((cloud_ply['R'], cloud_ply['G'], cloud_ply['B'])).T
        intensities = cloud_ply['intensity']

        # Parameters
        radius = 0.3
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
        filename = 'bunny_original.ply'
        name = filename.split('.')[0]
        cloud_path = os.path.join(data_path, filename)
        cloud_ply = read_ply(cloud_path)
        
        cloud_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        colors = np.vstack((cloud_ply['R'], cloud_ply['G'], cloud_ply['B'])).T
        intensities = cloud_ply['intensity']
        
        # Load points of interest
        query_points_indices = np.load(os.path.join(data_path, 'query_indices_{}.npy'.format(name)))

        # Parameters
        radius = 0.3
        query_points_indices = np.array([1,10,100])
        
        # Computations
        query_cov = compute_features(query_points_indices, cloud_points, colors, intensities, radius)
        
        # Save results
        np.save(os.path.join(saved_data_path, 'covariance_{}.npy'.format(name)), query_cov)
        
        #write_ply('../data/Lille_street_small_featured_{}.ply'.format(radius), [new_cloud], ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])


