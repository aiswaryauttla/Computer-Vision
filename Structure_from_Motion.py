import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple, Union, Dict


t_points = np.array
t_camera_parameters = np.array
t_descriptors = np.array
t_homography = np.array
t_view = np.array
t_img = np.array
t_images = Dict[str, t_img]
t_homographies = Dict[Tuple[str, str], t_homography]  # The keys are the keys of src and destination images

np.set_printoptions(edgeitems=30, linewidth=180,
                    formatter=dict(float=lambda x: "%8.05f" % x))


def fast_filter_and_align_descriptors(src: Tuple[t_points, t_descriptors], dst: Tuple[t_points, t_descriptors],
                                      similarity_threshold=0.8):
    """
    Aligns pairs of keypoints from two images.

    Aligns keypoints from two images based on descriptor similarity using Flann based matcher from cv2
    If K points have been detected in image1 and J points have been detected in image2, the result will be to sets of N
    points representing points with similar descriptors; where N <= J and K <=points.
    Args:
        src: A tuple of two numpy arrays with the first array having dimensions [N x 2] and the second one [N x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        dst: A tuple of two numpy arrays with the first array having dimensions [J x 2] and the second one [J x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        similarity_threshold: The ratio the distance of most similar descriptor to the distance of the second
            most similar.

    Returns:
        A tuple of numpy arrays both sized [N x 2] representing the similar point locations.
    """
    epsilon = .000000001 # for division by zero
    (points1, descriptors1), (points2, descriptors2) = src, dst

    FLANN_INDEX_LSH = 6
    flann_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=3)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    knnmatches = matcher.knnMatch(descriptors1, descriptors2, 2)

    first_to_second_ratios = np.zeros(len(knnmatches))
    indexes1 = np.zeros(len(knnmatches), dtype=np.int32)
    indexes2 = np.zeros(len(knnmatches), dtype=np.int32)
    for n, (first, second) in enumerate(knnmatches):
        first_to_second_ratios[n] = (first.distance / (second.distance + epsilon))
        indexes1[n] = first.queryIdx
        indexes2[n] = first.trainIdx

    keep_idx = first_to_second_ratios <= similarity_threshold
    filtered_indexes1, filtered_indexes2 = indexes1[keep_idx], indexes2[keep_idx]
    return points1[filtered_indexes1, :], points2[filtered_indexes2, :]



def inliers_epipolar_constraint(points1: t_points, points2: t_points, F:t_homography, distance_threshold: float) -> bool:
    """Returns indexes of points that satisfy the epipolar constraint

    Args:
        p1: A numpy array of shape (2, ) representing a single point from image 1
        p2: A numpy array of shape (2, ) representing a single point from image 2
        F: A numpy array of shape (3, 3) representing the fundamental matrix
        distance_threshold: a float representing the norm of the difference between to points so that they will be
            considered the same (near enough).

    Returns:
        An array of indexes of the points satisfy epipolar constraint.
    """
    ## TODO 2.1 Distance from the Epipolar Line
    # Step 1 - Transform p1 to other image 
    # First Convert points1 and points2 from 2D to (2+1)D homogenous coordinates - [x, y] --> [x, y, 1]
    src_3d = np.ones((points1.shape[0],3))
    dst_3d = np.ones((points2.shape[0],3))
    src_3d[:,:2] = points1
    dst_3d[:,:2] = points2
    src_3d = src_3d.T
    #dst_3d = dst_3d.T
    index=[]
    projected_points = np.dot(F, src_3d)
    # Step 2 - normalize the epipolar line 
    for i in range(points1.shape[0]):
        projected_points[:,i]=projected_points[:,i]/np.sqrt(projected_points[0,i]**2+projected_points[1,i]**2)
    # Step 3 - compute the point line distance (1 line)
        d=np.dot(dst_3d[i,:],projected_points[:,i])
        if abs(d)<distance_threshold:
            index.append(i)


    # Step 4 - check if the distance is smaller than distance threshold and return (2 lines)
        if abs(d) < distance_threshold:
            index.append(i)
    index = np.array(index)
    return index


def compute_fundamental_matrix(points1: t_points, points2: t_points) -> np.array:
    """

    Args:
        points1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.

    Returns:
        A [3 x 3] numpy array containing normalised fundamental matrix.
    """
    assert (len(points1) == 8), "Length of points1 should be 8!"
    assert (len(points2) == 8), "Length of points2 should be 8!"

    fundamental_matrix = np.zeros((8, 9)).astype('int')

    # TODO 2.2 8-Point Algorithm
    # Step 1 - Construct the 8x9 matrix A.
    A = np.zeros((points1.shape[0], 9))
    j = 0
    i=0
    while i<8:
        A[j, :] = np.array([points1[i][0]*points2[i][0], points1[i][0]*points2[i][1], points1[i][0], points1[i][1]*points2[i][0], points1[i][1]*points2[i][1], points1[i][1], points2[i][0], points2[i][1], 1])
        A[j + 1, :] = np.array([points1[i+1][0]*points2[i+1][0], points1[i+1][0]*points2[i+1][1], points1[i+1][0], points1[i+1][1]*points2[i+1][0], points1[i+1][1]*points2[i+1][1], points1[i+1][1],points2[i+1][0], points2[i+1][1], 1])
        j = j + 2
        i=i+2
    # Step 2 - Solve Af = 0 and extract F using SVD
    u, s, vt = np.linalg.svd(A)

    v = vt.T
    f = v[:, 8]
    F = [[f[0], f[3], f[6]], [f[1], f[4], f[7]], [f[2], f[5], f[8]]]
    # Step 3 - Enforce Rank(F) = 2
    # - Compute the SVD of F
    u, s, vt = np.linalg.svd(F)
    # - Set sigma3 to 0  (Hint: The singular values are stored as a vector in svd.w)
    s=np.diag(s)
    s[2,2]=0

    # - Recompute F with the updated sigma
    # - Normalize F and store in fundamental matrix
    F=np.dot(u,s)
    F=np.dot(F,vt)
    fundamental_matrix = F/F[2,2]
    return fundamental_matrix


def compute_F_ransac(points1: t_points, points2: t_points, distance_threshold: float=4.0, steps: int=1000,
                     n_points: int=8) -> Tuple[np.array, np.array]:
    """

    Args:
        points1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.
        distance_threshold: a float representing the norm of the difference between to points so that they will be
            considered the same (near enough)
        steps: An integer value representing steps for ransac optimization
        n_points: An integer value representing how many points ransac algorithm to implement

    Returns:
        best_homography: A numpy array of size [3 x 3] representing the best/optimized homography
        (having maximum inliers)
    """
    best_count = 0
    best_homography = np.eye(3)
    best_inlier_indexes = np.array([])

    for n in tqdm(range(steps)):
        if n == steps - 1:
            print(f"Step: {n:4}  {best_count} RANSAC points match!")

        randomidx = np.random.permutation(points1.shape[0])[:n_points]
        rnd_points1, rnd_points2 = points1[randomidx, :], points2[randomidx, :]

        homography = compute_fundamental_matrix(rnd_points1, rnd_points2)
        inliers_indexes = inliers_epipolar_constraint(points1, points2, homography, distance_threshold)

        if inliers_indexes.shape[0] > best_count:
            best_count = inliers_indexes.shape[0]
            best_homography = homography
            best_inlier_indexes = inliers_indexes
    return best_homography, np.array(best_inlier_indexes)



def triangulate(P1: t_view, P2: t_view, p1: t_points, p2: t_points) -> np.array:
    """

    Args:
        P1: A [3 x 4] numpy array representing the camera matrix for View 1 (camera 1 is at 0, 0, 0)
        P2: A [3 x 4] numpy array representing the camera matrix for View 2
        p1: A numpy array of shape (2, ) representing a single point from image 1
        p2: A numpy array of shape (2, ) representing a single point from image 2
    Returns:
        resulting_point: A numpy array of shape (3, ) representing the point in 3D space
    """
    epsilon = .00000001 # avoiding division by zero
    # TODO 3 Triangulation
    # Step 1 - Construct the matrix A (~5 lines)
    # Hint - homogenous solution - Zisserman Book page 312
    A=np.ones((4,4))

    A[0,:]=p1[0]*P1[2,:]-P1[0,:]
    A[1,:]=p1[1]*P1[2,:]-P1[1,:]
    A[2,:]=p2[0]*P2[2,:]-P2[0,:]
    A[3,:]=p2[1]*P2[2,:]-P2[1,:]

    # Step 2 - Extract the solution and project it back to real 3D (from homogenous space) 
    u, s, vt = np.linalg.svd(A)
    v = vt.T
    h = v[:,3]
    h=h[0:3]/(h[3]+epsilon)

    return h

def triangulate_all_points(View1: t_view, View2: t_view, K: t_view, points1: t_points, points2: t_points) \
        -> Tuple[np.array, np.array]:
    """

    Args:
        View1: A numpy array of shape [3 x 4] representing View matrix 1 for Camera
        View2: A numpy array of shape [3 x 4] representing View matrix 2 for Camera
        K: A numpy array of shape [3 x 3] representing the intrinsic camera parameters
        points1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.

    Returns:
        wps: A numpy array of shape [N x 3] representing the point cloud in 3D space
        infront_of_camera: A numpy array of shape (N, ) containing the boolean values for all the points from wps
            informing whether that point is in front of camera or not.
    """
    wps = []
    P1 = np.dot(K, View1)
    P2 = np.dot(K, View2)
    infront_of_camera = np.ones(len(points1), dtype=np.bool)
    for i in range(len(points1)):
        wp = triangulate(P1, P2, points1[i], points2[i])
        # Check if this points is in front of both cameras
        ptest = [wp[0], wp[1], wp[2], 1]
        p1 = np.matmul(P1, ptest)
        p2 = np.matmul(P2, ptest)
        wps.append(wp)
        infront_of_camera[i] = p1[2] > 0 and p2[2] > 0
    wps = np.array(wps)
    return wps, infront_of_camera


def compute_essential_matrix(fundamental_matrix: t_homography, camera_parameters: t_camera_parameters) -> t_homography:
    """
    Args:
        fundamental_matrix: A [3 x 3] numpy array representing the fundamental matrix
        camera_parameters: A [3 x 3] numpy array representing the instrinsic camera parameters

    Returns:
        essential_matrix: A [3 x 3] numpy array representing the essential matrix
    """
    # TODO 4.1: Calculate essential matrix (3 lines)

    essential_matrix=camera_parameters.T@fundamental_matrix@camera_parameters
    essential_matrix=essential_matrix/essential_matrix[2,2]
    return essential_matrix


def decompose(E: t_homography) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Args:
        E: A [3 x 3] numpy array representing the essential matrix

    Returns:
        R1, R2: 2 rotation matrices of shape [3 x 3]
        t1, t2: 2 translation matrices of shape (3, )
    """
    # TODO 4.2 Calculating Rotation and translation matrices

    # Step 1 - Compute the SVD E (~1 line)
    u, s, vt = np.linalg.svd(E)
    # Step 2 - Compute W and Possible rotations (~3-6 lines)
    W=np.ones((3,3))
    W[0,:]=[0,1,0]
    W[1,:]=[1,0,0]
    W[2,:]=[0,0,1]
    R1=u@W@vt
    R2=u@W.T@vt

    # Step 3 - Possible translations and return R1, R2, t1, t2 (~4 lines)
    S=u[:,2]/np.linalg.norm(u[:,2], ord=1)

    t1=S

    t2=-S
    return R1,R2,t1,t2

def relativeTransformation(E: t_homography, points1: t_points, points2: t_points, K: t_camera_parameters) -> t_view:
    """

    Args:
        E: A [3 x 3] numpy array representing the essential matrix
        points1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        points2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.
        K: A numpy array of shape [3 x 3] representing the intrinsic camera parameters

    Returns:
        V: A numpy array of shape [3 x 4] representing View matrix 2 for Camera
    """

    R1, R2, t1, t2 = decompose(E)
    ## A negative determinant means that R contains a reflection.This is not rigid transformation!
    if np.linalg.det(R1) < 0:
        E = -E
        R1, R2, t1, t2 = decompose(E)

    bestCount = 0

    for dR in range(2):
        if dR == 0:
            cR = R1
        else:
            cR = R2
        for dt in range(2):
            if dt == 0:
                ct = t1
            else:
                ct = t2

            View1 = np.eye(3, 4)
            View2 = np.zeros((3, 4))
            for i in range(3):
                for j in range(3):
                    View2[i, j] = cR[i, j]
            for i in range(3):
                View2[i, 3] = ct[i]

            count = len(triangulate_all_points(View1, View2, K, points1, points2))
            if (count > bestCount):
                V = View2

    return V










