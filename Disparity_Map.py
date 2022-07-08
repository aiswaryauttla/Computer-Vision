import cv2
import numpy as np
import scipy.ndimage
import scipy.signal
from ex2 import extract_features, filter_and_align_descriptors
import math

t_img = np.array
t_disparity = np.array
from scipy.spatial import distance

def get_max_translation(src: t_img, dst: t_img, well_aligned_thr=.1) -> int:
    """finds the maximum translation/shift between two images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        well_aligned_thr: a float representing the maximum y wise distance between valid matching points.

    Returns:
        an integer value representing the maximum translation of the camera from src to dst image
    """

    # Step 1: Generate features/descriptors, filter and align them 

    t_points,t_descriptors=extract_features(src)
    f1=(t_points,t_descriptors)
    t_points, t_descriptors = extract_features(dst)
    f2 = (t_points, t_descriptors)
    t_points_f1,t_points_f2=filter_and_align_descriptors(f1,f2)
    # Step 2: filter out correspondences that are not horizontally aligned using well aligned threshold
    dist=np.zeros(t_points_f1.shape[0])

    for i in range(t_points_f1.shape[0]):

        dist[i]=np.linalg.norm(t_points_f1[i][1]-t_points_f2[i][1])
    d=np.where(dist<=well_aligned_thr)
    d = np.asarray(d)
    valid_t_points_f1=t_points_f1[d].reshape(d.shape[1],2)
    valid_t_points_f2=t_points_f2[d].reshape(d.shape[1],2)
    #d=d.T
    # Step 3: Find the translation across the image using the descriptors and return the maximum value
    diff=valid_t_points_f2[:,0]-valid_t_points_f1[:,0]
    maximum=math.ceil(np.max(diff))
    minimum=math.floor(np.min(diff))
    if abs(minimum)>abs(maximum):
        translation=minimum
    else:
        translation=maximum

    return translation


def render_disparity_hypothesis(src: t_img, dst: t_img, offset: int, pad_size: int) -> t_disparity:
    """Calculates the agreement between the shifted src image and the dst image.

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation

    Returns:
        a numpy array of shape [H x W] containing the euclidean distance between RGB values of the shifted src and dst
        images.
    """

    # Step 1: Pad necessary values to src and dst

    src_padded=np.pad(src,[(0,0),(offset,pad_size-offset),(0,0)])
    dst_padded=np.pad(dst,[(0,0),(0,pad_size),(0,0)])
    # Step 2: find the disparity value and return
    src_shifted=np.zeros_like(src_padded)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src_shifted[i,j-offset,:]=src[i,j,:]

    disp=dst_padded-src_shifted
    disp=disp**2
    disp=np.sum(disp,axis=2)
    disparity=np.sqrt(disp)



    return disparity


def disparity_map(src: t_img, dst: t_img, offset: int, pad_size: int, sigma_x: int, sigma_z: int,
                  median_filter_size: int) -> t_disparity:
    """calculates the best/minimum disparity map for a given pair of images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation
        sigma_x: an integer value for standard deviation in x-direction for gaussian filter
        sigma_z: an integer value for standard deviation in z-direction for gaussian filter
        median_filter_size: an integer value representing the window size for applying median filter

    Returns:
        a numpy array of shape [H x W] containing the minimum/best disparity values for a pair of images
    """

    # Step 1: Construct a stack of all reasonable disparity hypotheses.
    im = np.zeros((src.shape[0], src.shape[1]+pad_size,pad_size+1))

    for i in range(offset+1):
        im[:,:,i]= render_disparity_hypothesis(src, dst, i, pad_size)

    # Step 2: Enforce the coherence between x-axis and disparity-axis using a 3D gaussian filter onto
    # the stack of disparity hypotheses

    filtered_disparity=scipy.ndimage.gaussian_filter(im,(0,sigma_x,sigma_z),0)
    # Step 3: Choose the best disparity hypothesis for every pixel
    minimum_disparity = np.argmin(filtered_disparity,axis=2)


    # Step 4: Apply the median filter to enhance local consensus
    disparity=scipy.ndimage.median_filter(minimum_disparity, size=median_filter_size)
    return disparity


def bilinear_grid_sample(img: t_img, x_array: t_img, y_array: t_img) -> t_img:
    """Sample an image according to a sampling vector field.

    Args:
        img: one image, numpy array of shape [H x W x 3]
        x_array: a numpy array of [H' x W'] representing the x coordinates src x-direction
        y_array: a numpy array of [H' x W'] representing interpolation in y-direction

    Returns:
        An image of size [H' x W'] containing the sampled points in
    """

    # Step 1: Estimate the left, top, right, bottom integer parts (l, r, t, b)
    # and the corresponding coefficients (a, b, 1-a, 1-b) of each pixel
    left=np.arange(img.shape[1])
    left=np.tile(left, (img.shape[0], 1))
    right=left+1
    top=np.arange(img.shape[0])
    top= np.transpose([top] * img.shape[1])
    bottom=top+1
    a=x_array-left
    b=y_array-top
    c=1-a
    d=1-b

    # Step 2: Take care of out of image coordinates
    a[a>1]=0
    a[a<0]=0
    b[b>1]=0
    b[b<0]=0
    c=1-a
    d=1-b
    img=np.pad(img,[(0,1),(0,1),(0,0)])
    # Step 3: Produce a weighted sum of each rounded corner of the pixel
    output=np.zeros((img.shape[0]-1,img.shape[1]-1,3))
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            output[i][j][:]=(c[i][j]*d[i][j]*np.round(img[top[i][j]][left[i][j]][:]))+(a[i][j]*d[i][j]*np.round(img[top[i][j]][right[i][j]][:]))+np.round((c[i][j]*b[i][j]*img[bottom[i][j]][left[i][j]][:]))+(a[i][j]*b[i][j]*np.round(img[bottom[i][j]][right[i][j]][:]))



    # Step 4: Accumulate and return all the weighted four corners
    #output=np.sum(output,axis=2)
    return output

#
