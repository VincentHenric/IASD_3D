#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # YOUR CODE
    neighborhoods = []
    for query in queries:
        neighborhood = supports[np.where(np.linalg.norm(supports-query, axis=1)<=radius**2)]
        neighborhoods.append(neighborhood)
        
    #neighborhoods = np.concatenate(neighborhoods, axis=0)

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = []
    for query in queries:
        neighborhood_index  = np.argpartition(np.linalg.norm(supports-query, axis=1), kth=k+1, axis=0)[:k+1]
        neighborhood = supports[neighborhood_index[1:]]
        neighborhoods.append(neighborhood)

    return neighborhoods

def brute_force_KNN_2(queries, supports, k):

    # YOUR CODE
    neighborhoods = []
    for query in queries:
        neighborhood_index  = np.argsort(np.linalg.norm(supports-query, axis=1), axis=0)[:k+1]
        neighborhood = supports[neighborhood_index[1:]]
        neighborhoods.append(neighborhood)

    return neighborhoods

def kdtree_neighbors(queries, supports, leaf_size, radius):
    t0 = time.time()
    kdtree = KDTree(supports, leaf_size)
    t1 = time.time()
    neighborhood_indices = kdtree.query_radius(queries, radius, count_only=False, return_distance=False)
    neighborhoods = [supports[neighborhood_index] for neighborhood_index in neighborhood_indices]
    t2 = time.time()
    return neighborhoods, t1-t0, t2-t1



# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #
    
    flag_brute_force = False
    flag_leaf_size = False
    flag_radius = True

    # If statement to skip this part if you want
    if flag_brute_force:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if flag_leaf_size:

        # Define the search parameters
        num_queries = 1000
        num_iterations = 5
        radius = 0.2

        # YOUR CODE
        #leaf_sizes = np.linspace(40,60,20, dtype=int)
        leaf_sizes = np.sort(np.concatenate([np.linspace(10,1000,20, dtype=int), np.linspace(40,60,20, dtype=int)]))
        times = np.zeros((len(leaf_sizes), num_iterations, 2))
        
        
        for j in range(num_iterations):
            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]
            
            for i, leaf_size in enumerate(leaf_sizes):
                # launch neighborhood search
                neighborhoods, d1, d2 = kdtree_neighbors(queries, points, leaf_size, radius)
                times[i,j,0] = d1
                times[i,j,1] = d2
        
                # Print timing results
                print('{:d} kdtree of leaf size {} computed in {:.3f} seconds for iteration {}'.format(num_queries, leaf_size, d2, j))
            
        plt.plot(leaf_sizes, times[:,:,1].mean(axis=1))
        plt.xlabel('leaf size')
        plt.ylabel('time for queries')
        plt.savefig('../images/time_queries_vs_leaf_size.png')
        plt.show()
        
        plt.plot(leaf_sizes, times[:,:,0].mean(axis=1))
        plt.xlabel('leaf size')
        plt.ylabel('time for kd tree creation')
        plt.savefig('../images/time_kd_tree_vs_leaf_size.png')
        plt.show()
        
        opt_leaf_size = leaf_sizes[np.argmin(times[:,:,1].mean(axis=1))]
        print('Optimal leaf size: {}'.format(opt_leaf_size))
        
    if flag_radius:
        # Define the search parameters
        num_queries = 1000
        num_iterations = 1
        radius = 0.2
        
        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        
#        # YOUR CODE
        leaf_size = 54 # the optimal value
        radiuses = [0.1,0.2,0.4,0.6,0.8,1,1.2,1.4]
        times = np.zeros((len(radiuses), num_iterations, 2))
    
        for j in range(num_iterations):
            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]
            
            for i, radius in enumerate(radiuses):
                # launch neighborhood search
                neighborhoods, d1, d2 = kdtree_neighbors(queries, points, leaf_size, radius)
                times[i,j,0] = d1
                times[i,j,1] = d2
            
                # Print timing results
                print('{:d} kdtree of radius {} computed in {:.3f} seconds for iteration {}'.format(num_queries, radius, d2, j))
        
        plt.plot(radiuses, times[:,:,1].mean(axis=1))
        plt.xlabel('radius')
        plt.ylabel('time for queries')
        plt.savefig('../images/time_queries_vs_radius.png')
        plt.show()
        
        plt.plot(radiuses, times[:,:,0].mean(axis=1))
        plt.xlabel('radius')
        plt.ylabel('time for kd tree creation')
        plt.savefig('../images/time_kd_tree_vs_radius.png')
        plt.show()
        
        
        
        
        
        i = radiuses.index(0.2)
        d1,d2 = times.mean(axis=1)[i,0], times.mean(axis=1)[i,1]
        
        # Print timing results
        print('{:d} kd tree search computed in {:.3f} seconds'.format(num_queries, d2))

        # Time to compute all neighborhoods in the cloud
        total_tree_search_time = points.shape[0] * d2 / num_queries
        print('Computing kd tree search on whole cloud : {:.0f} hours'.format(total_tree_search_time  / 3600))



        #s3dis dataset