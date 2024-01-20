import cv2
import numpy as np

def Affine_trans(prev,curr):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # prev_gray=prev.copy()
    # curr_gray=curr.copy()


    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=500,
                                       qualityLevel=0.1,
                                       minDistance=5,
                                       blockSize=3)
    
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None)
    
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    #Find transformation matrix

    [m, inliers] = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    M_matrix = np.vstack((m, np.array([0, 0, 1])))

    return M_matrix
    
    

    