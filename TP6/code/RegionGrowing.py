#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Plane detection by region growing
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

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


def compute_features(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud
    epsilon = 1e-16
    all_eigenvalues, all_eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)
    normal_vectors = all_eigenvectors[:,0]
    normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=-1)[:,None]
    
    #verticality = 2 * np.arcsin(np.abs(normal_vectors[:,-1]))/np.pi
    #linearity = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2]+epsilon)
    planarity = (all_eigenvalues[:,1]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+epsilon)
    #sphericity = all_eigenvalues[:,0]/(all_eigenvalues[:,2]+epsilon)

    return planarity, normal_vectors



def compute_planarities_and_normals(points, radius):

    #normals = points * 0
    #planarities = points[:, 0] * 0

    # TODO:
    planarities, normals = compute_features(points, points, radius)

    return planarities, normals


def region_criterion(p1, p2, n1, n2, threshold1=0.05, threshold2=0.05):
    return np.logical_and(np.abs((p2-p1)@n1) < threshold1, np.arccos(np.clip(np.abs(n2@n1),0,1))< threshold2)


def queue_criterion(p, threshold=0.9):
    return p>threshold


def RegionGrowing(cloud, normals, planarities, radius):

    # TODO:
    

    N = len(cloud)
    region = np.zeros(N, dtype=bool)
    inspected = np.zeros(N, dtype=bool)
    
    i = 0
    c = 0
    seed = np.random.randint(N)
    
    queue = [seed]
    region[seed] = 1
    inspected[seed] = 1
    
    while queue != []:
        if i%1000 == 0:
            print('Iteration {}'.format(i))
            print('{} elements'.format(c))
        q_index = queue.pop()
        queue_pt = cloud[[q_index]]
        neighbors_mask = np.linalg.norm(cloud-queue_pt, axis=1)<radius
        neighbors_index = np.where(neighbors_mask)[0]
        selected_neighbors_mask = region_criterion(queue_pt,
                                                   cloud[neighbors_mask],
                                                   normals[q_index],
                                                   normals[neighbors_mask],
                                                   threshold1=0.1,
                                                   threshold2=0.1)
        
        admitted_indices = neighbors_index[selected_neighbors_mask]
        admitted_mask = np.zeros(N, dtype=bool)
        admitted_mask[admitted_indices] = True
        new_indices = np.where(~inspected & admitted_mask)[0]
        
        region[admitted_indices] = 1
        inspected[admitted_indices] = 1
        
        for idx in new_indices:
            if queue_criterion(planarities[idx], threshold=0.7):
                queue.append(idx)
                c += 1
        
        i+=1

    return region


def multi_RegionGrowing(cloud, normals, planarities, radius, NB_PLANES=2):

    # TODO:

    plane_inds = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_inds = np.arange(0, N)

    return plane_inds, remaining_inds, plane_labels


# ----------------------------------------------------------------------------------------------------------------------
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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    N = len(points)

    # Computes normals of the whole cloud
    # ***********************************
    #
    if False:
        # Parameters for normals computation
        radius = 0.2
    
        # Computes normals of the whole cloud
        t0 = time.time()
        planarities, normals = compute_planarities_and_normals(points, radius)
        t1 = time.time()
        print('normals and planarities computation done in {:.3f} seconds'.format(t1 - t0))
    
        # Save
        write_ply('../planarities.ply',
                  [points, planarities],
                  ['x', 'y', 'z', 'planarities'])

    # Find a plane by Region Growing
    # ******************************
    #

    if True:
        # Load point cloud
        data = read_ply('../planarities.ply')
    
        # Concatenate data
        points = np.vstack((data['x'], data['y'], data['z'])).T
        planarities = data['planarities'].T
        N = len(points)
    
        # Define parameters of Region Growing
        radius = 0.2

        # Find a plane by Region Growing
        t0 = time.time()
        region = RegionGrowing(points, normals, planarities, radius)
        t1 = time.time()
        print('Region Growing done in {:.3f} seconds'.format(t1 - t0))

        # Get inds from bollean array
        plane_inds = region.nonzero()[0]
        remaining_inds = (1 - region).nonzero()[0]

        # Save the best plane
        write_ply('../best_plane_regiongrowing.ply',
                  [points[plane_inds], colors[plane_inds], region[plane_inds].astype(np.int32), planarities[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'planarities'])
        write_ply('../remaining_points_regiongrowing.ply',
                  [points[remaining_inds], colors[remaining_inds], region[remaining_inds].astype(np.int32), planarities[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'planarities'])

    # Find multiple in the cloud
    # ******************************
    #

    if False:
        # Define parameters of multi_RANSAC
        radius = 0.2
        NB_PLANES = 10

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RegionGrowing(points, normals, planarities, radius, NB_PLANES)
        t1 = time.time()
        print('multi RegionGrowing done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
