import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute dx and dy with cv2.Sobel. 
    Idx=cv2.Sobel(I,ddepth=cv2.CV_32F,dx=1,dy=0)
    Idy=cv2.Sobel(I,ddepth=cv2.CV_32F,dx=0,dy=1)

    # Step 2: Ixx Iyy Ixy from Idx and Idy 
    Ixx=Idx*Idx
    Iyy=Idy*Idy
    Ixy=Idx*Idy


    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur 
    A=cv2.GaussianBlur(Ixx,(3,3),1)
    B=cv2.GaussianBlur(Iyy,(3,3),1)
    C=cv2.GaussianBlur(Ixy,(3,3),1)
    T=np.array([[A,C],[C,B]])
    #R=(np.dot(A,B)-np.dot(C,C))-k*((A+B)**2)


    #Step 4:  Compute the harris response with the determinant and the trace of T 
    det_T=A*B-C*C
    trace_T=A+B
    R = det_T-k*trace_T**2

    return R,A,B,C,Idx,Idy

def detect_corners(R: np.array, threshold: float = 0.01) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization 
    #empty=np.zeros((1,2))
    y_final=np.array([])
    x_final=np.array([])
    image = np.pad(R,1,mode='constant')
    # Step 2 (recommended) : create one image for every offset in the 3x3 neighborhood .

    for x in range(1,image.shape[1]):
        for y in range(1,image.shape[0]):
            if image[y,x]>threshold:
                img=image[y-1:y+2,x-1:x+2]



    # Step 3 (recommended) : compute the greatest neighbor of every pixel 
                (_,_,_,maxLoc)=cv2.minMaxLoc(img)

                if maxLoc==(1,1):
                    y_final=np.append(y_final,y)
                    x_final=np.append(x_final,x)
                    #empty = np.vstack((empty,(y,x)))

    #empty=np.delete(empty, 0, 0)
    # Step 4 (recommended) : Compute a boolean image with only all key-points set to True 


    # Step 5 (recommended) : Use np.nonzero to compute the locations of the key-points from the boolean image 

    return (x_final,y_final)


def detect_edges(R: np.array, edge_threshold: float = -0.01, epsilon=-.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization 
    empty = np.zeros((1, 2))
    y_final = np.array([])
    x_final = np.array([])
    image = np.pad(R, 1, mode='constant')
    # Step 2 (recommended) : Calculate significant response pixels 
    for y in range(1,image.shape[0]):
        for x in range(1,image.shape[1]):
            if image[y,x]<edge_threshold:


    # Step 3 (recommended) : create two images with the smaller x-axis and y-axis neighbors respectively .
                x_image = image[y,x-1:x:x + 2]
                y_image = image[y-1:y:y+2,x]

    # Step 4 (recommended) : Calculate pixels that are lower than either their x-axis or y-axis neighbors 
                (_, _, minLoc_1,_) = cv2.minMaxLoc(x_image)
                (_, _, minLoc_2, _) = cv2.minMaxLoc(y_image)

    # Step 5 (recommended) : Calculate valid edge pixels by combining significant and axis_minimal pixels 
                if minLoc_1==minLoc_2:
                    if minLoc_1==(0,0):
                        y_final = np.append(y_final,int(y))
                        x_final = np.append(x_final, int(x))

    p = x_final.astype(int)
    q=y_final.astype(int)
    for i in range(len(p)):
        R[q[i],p[i]]=1
    R[R!=1]=0
    R=R==1
    return R
