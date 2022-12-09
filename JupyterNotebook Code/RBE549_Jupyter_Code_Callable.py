import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import time
import sys
import PySimpleGUI as sg

from lib.visualization.plotting import visualize_paths
from lib.visualization.video import play_trip
import stereo_visual_odometry as VO

from tqdm import tqdm

def main(samples,animate,stereo,feature,optical):
    """
    Calculates the average processing time and error of a 50 frame KITTI test sample

    Parameters
    ----------
    samples (int):      Number of time to cycle through code
    animate ([0,1]):    0 - No animation
                        1 - Run with animation
    stereo ([1,2]):     1 - StereoSGBM_create
                        2 - StereoBM_create
    feature ([1,2,3]):  1 - FastFeatureDetector_create
                        2 - SIFT_create
                        3 - ORB_create
    optical ([1,2]):    1 - calcOpticalFlowPyrLK
                        2 - calcOpticalFlowSparseRLOF

    Returns
    -------
    No data is returned. Averages and configuration are automaticlly printed
    """

    stereo_options = [0,'StereoSGBM_create','StereoBM_create']
    feature_options = [0,'FastFeatureDetector_create','SIFT_create','ORB_create' ]
    optical_options = [0,'calcOpticalFlowPyrLK','calcOpticalFlowSparseRLOF']

    global P_l, K_l, P_r, K_r 

    data_dir = 'KITTI_sequence_2'  # Try KITTI_sequence_2

    with open(data_dir + '/calib.txt', 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P_l = np.reshape(params, (3, 4))
        K_l = P_l[0:3, 0:3]
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P_r = np.reshape(params, (3, 4))
        K_r = P_r[0:3, 0:3]

    gt_poses = []
    with open(data_dir + '/poses.txt', 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            gt_poses.append(T)
    
    global images_l, images_r

    image_paths = [os.path.join(data_dir + '/image_l', file) for file in sorted(os.listdir(data_dir + '/image_l'))]
    images_l = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    image_paths = [os.path.join(data_dir + '/image_r', file) for file in sorted(os.listdir(data_dir + '/image_r'))]
    images_r = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    global disparity, disparities

    if stereo == 1:
        # For explication of rows 20 - 31 use https://www.andreasjakl.com/how-to-apply-stereo-matching-to-generate-depth-maps-part-3/ and https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html
        block = 15

        SGBM_params = dict(minDisparity = 0, 
                            numDisparities = 256, 
                            blockSize = block,
                            uniquenessRatio = 5, 
                            speckleWindowSize = 200, 
                            speckleRange = 2, 
                            disp12MaxDiff = 0,
                            P1 = block * block * 8,
                            P2 = block * block * 32)
        
        disparity = cv2.StereoSGBM_create(**SGBM_params)

        disparities = [np.divide(disparity.compute(images_l[0], images_r[0]).astype(np.float32), 16)]
        
    elif stereo == 2:

        BM_params = dict(numDisparities = 16*5, 
                                blockSize = 9)

        disparity = cv2.StereoBM_create(**BM_params)

        disparities = [np.divide(disparity.compute(images_l[0], images_r[0]).astype(np.float32), 16)]
    else:
        test_layout = [[sg.Text('Invalid Stereo Input. Only 1 and 2 are valid         ')],
            [sg.Button('Exit')]]

        test_window2 = sg.Window('Error Message', test_layout)
        while True:
            event, values = test_window2.read()
            if event in (None, 'Exit'):
                test_window2.Close()
                sys.exit()   
    
    global fastFeatures

    if feature == 1:
        # This article could be a good source to improve feature detecion: https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
        fastFeatures = cv2.FastFeatureDetector_create()

    elif feature == 2:
        fastFeatures = cv2.SIFT_create()

    elif feature == 3:
        fastFeatures = cv2.ORB_create()

    else:
        test_layout = [[sg.Text('Invalid feature Input. Only 1, 2, and 3 are valid         ')],
            [sg.Button('Exit')]]

        test_window2 = sg.Window('Error Message', test_layout)
        while True:
            event, values = test_window2.read()
            if event in (None, 'Exit'):
                test_window2.Close()
                sys.exit()
    
    global opti_params

    if optical==1:
        # LK
        opti_params = dict(winSize=(15, 15),
                        flags=cv2.MOTION_AFFINE,
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
    elif optical==2: 
        test_layout = [[sg.Text('Option still under construction.             \nDo you want to continue?')],
            [sg.Button('Continue'),sg.Button('Exit')]]

        test_window2 = sg.Window('Warning Message', test_layout)

        while True:
            event, values = test_window2.read()
            if event in (None, 'Exit'):
                test_window2.Close()
                sys.exit() 
            if event in (None, 'Continue'):
                test_window2.Close()
                
                # RLOF
                opti_params = cv2.optflow.RLOFOpticalFlowParameter_create()
                opti_params.setMaxIteration(30)
                opti_params.setNormSigma0(3.2)
                opti_params.setNormSigma1(7.0)
                opti_params.setLargeWinSize(21)
                opti_params.setSmallWinSize(9)
                opti_params.setMaxLevel(9)
                opti_params.setMinEigenValue(0.0001)
                opti_params.setCrossSegmentationThreshold(25)
                opti_params.setGlobalMotionRansacThreshold(10.0)
                break 

        
        

    else:
        test_layout = [[sg.Text('Invalid Optical Input. Only 1 and 2 are valid         ')],
            [sg.Button('Exit')]]

        test_window2 = sg.Window('Error Message', test_layout)
        while True:
            event, values = test_window2.read()
            if event in (None, 'Exit'):
                test_window2.Close()
                sys.exit() 
      
    def track_keypoints(img1, img2, kp1, optical, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        if optical == 1:
            # Use optical flow to find tracked counterparts
            trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **opti_params)

        elif optical == 2:
            # Use optical flow to find tracked counterparts
            img1_cvt = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
            img2_cvt = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
            trackpoints2, st, err = cv2.optflow.calcOpticalFlowSparseRLOF(img1_cvt, img2_cvt, trackpoints1, None, rlofParam = opti_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2
    
    def calculate_right_qs(q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(P_l, P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(P_l, P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])

        return Q1, Q2

    def form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def reprojection_residuals(dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(P_l, transf)
        b_projection = np.matmul(P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        
        return residuals

    def estimate_pose(q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = form_transf(R, t)

        return transformation_matrix

    def get_pose(i, optical):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l, img2_l = images_l[i - 1:i + 1]

        # Get teh tiled keypoints
        #kp1_l = get_tiled_keypoints(img1_l, 10, 20)
        kp1_l = fastFeatures.detect(img1_l)
        
        # Track the keypoints
        tp1_l, tp2_l = track_keypoints(img1_l, img2_l, kp1_l, optical)
        
        # Calculate the disparitie
        disparities.append(np.divide(disparity.compute(img2_l, images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = calculate_right_qs(tp1_l, tp2_l, disparities[i - 1], disparities[i])

        # Calculate the 3D points
        Q1, Q2 = calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = estimate_pose(tp1_l, tp2_l, Q1, Q2)
        
        return transformation_matrix

    if animate == 1:
        play_trip(images_l, images_r)  # Comment out to not play the trip

    times = []
    errors = []
    for k in range(samples):
        gt_path = []
        estimated_path = []

        if animate == 1:
            poses = enumerate(tqdm(gt_poses, unit="poses"))
        else:
            poses = enumerate(gt_poses)

        start = time.time()
        for i, gt_pose in poses:
            if i < 1:
                cur_pose = gt_pose
            else:
                transf = get_pose(i, optical)
                cur_pose = np.matmul(cur_pose, transf)
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        end = time.time()

        ellapsed = end-start  
        times.append(ellapsed)
        errors.append(np.sqrt((gt_path[i][0]-estimated_path[i][0])**2+(gt_path[i][1]-estimated_path[i][1])**2))

        if animate == 1:
            visualize_paths(gt_path, estimated_path, 'Configuration: {}, {}, {}'.format(stereo_options[stereo],feature_options[feature],optical_options[optical]),
                                    file_out=os.path.basename(data_dir) + ".html")
            
            print(ellapsed)  
            
    print('Configuration: {}, {}, {}'.format(stereo_options[stereo],feature_options[feature],optical_options[optical]))
    print('Average Ellapsed Time: {}'.format(np.mean(times)))
    print('Average Error: {}'.format(np.mean(errors)))


#main(1,1,2,1,1)