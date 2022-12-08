import numpy as np
import cv2
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt
import os

def visualOdometry(Left_Images,Right_Images,Current_poseMatrix):
    """
    Calculates the average processing time and error of a 51 frame KITTI test sample

    Parameters
    ----------
    Left_Images ([Previous Image, Current Image]): Current and previous images from left camera
    Right_Images ([Previous Image, Current Image]): Current and previous images from left camera
    Current_poseMatrix ([4x4]): Current positional matrix as a 4x4 array

    Returns
    -------
    New_poseMatrix ([4x4]): New postional matrix as a 4x4 array
    New_Position ([X,Y]): Positional change in X and Y coordinates
    Ellapsed_Time (float): Time ellapsed during visual odometry capture for current positional change
    """

    global P_l, K_l, P_r, K_r 
    
    # KITTI Dataset Calibration
    '''
    P_l = np.array([[718.856, 0., 607.1928, 0.],[ 0., 718.856, 185.2157, 0.],[0., 0., 1., 0.]])
    K_l = P_l[0:3, 0:3]
    
    P_r = np.array([[718.856, 0., 607.1928, -386.1448],[ 0., 718.856, 185.2157, 0.],[0., 0., 1., 0.]])
    K_r = P_r[0:3, 0:3]
    '''
    # Real Time Code Calibration - Low Cost Camera
    '''
    P_l = np.array([[193.10259135106102, 0.0, 180.88180313102188, 0.0],[0.0, 193.60656458116262, 138.921210372036, 0.0],[0.0, 0.0, 1.0, 0.0]])
    K_l = P_l[0:3, 0:3]

    P_r = np.array([[201.85691796708062, -1.582924355677401, 171.66026338372794, -339.9667196110423],[-4.60581250972677, 208.5442719977075, 139.86812990881378, 93.85133321558763],[-0.03978103329420952, -0.0038134229457450377, 0.9992011445126961, 0.6095093046641432]])
    K_r = P_r[0:3, 0:3]
    '''
    '''
    P_l = np.array([[91.85033416748047, 0.0, 191.75708723068237, 0.0],[0.0, 91.85033416748047, 135.02482390403748, 0.0],[0.0, 0.0, 1.0, 0.0]])
    K_l = P_l[0:3, 0:3]

    P_r = np.array([[91.85033416748047, 0.0, 191.75708723068237, 5773.854999489385],[0.0, 91.85033416748047, 135.02482390403748, 0.0],[0.0, 0.0, 1.0, 0.0]])
    K_r = P_r[0:3, 0:3]
    '''
    # Real Time Code Calibration - Zed2 Camera
    
    P_l = np.array([[513.0909576327231, 0.0, 638.4887540503237, 0.0],[0.0, 516.4312524995516, 385.0964516155614, 0.0],[0.0, 0.0, 1.0, 0.0]])
    K_l = P_l[0:3, 0:3]

    P_r = np.array([[511.7983742817586, -8.689457794397656, 634.1373710175565, -2625.6463685768294],[-0.4090062051800656, 507.55207120877066, 374.12970612424056, -24.763317290172175],[0.00019204794305597482, -0.014455605593522164, 0.9998954938314865, -0.05696334883402989]])
    K_r = P_r[0:3, 0:3] 
    
    global images_l, images_r

    images_l = Left_Images
    images_r = Right_Images

    global disparity, disparities

    BM_params = dict(numDisparities = 16*5, 
                            blockSize = 9)

    disparity = cv2.StereoBM_create(**BM_params)

    disparities = [np.divide(disparity.compute(images_l[0], images_r[0]).astype(np.float32), 16)]
    
    global fastFeatures

    # This article could be a good source to improve feature detecion: https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
    fastFeatures = cv2.FastFeatureDetector_create()
    
    global opti_params

    # LK
    opti_params = dict(winSize=(15, 15),
                    flags=cv2.MOTION_AFFINE,
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
          
    def track_keypoints(img1, img2, kp1, max_error=4):
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

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **opti_params)

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

    def get_pose():
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
        img1_l, img2_l = images_l[0:2]

        # Get teh tiled keypoints
        #kp1_l = get_tiled_keypoints(img1_l, 10, 20)
        kp1_l = fastFeatures.detect(img1_l)
        
        # Track the keypoints
        tp1_l, tp2_l = track_keypoints(img1_l, img2_l, kp1_l)
        
        # Calculate the disparitie
        disparities.append(np.divide(disparity.compute(img2_l, images_r[1]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = calculate_right_qs(tp1_l, tp2_l, disparities[0], disparities[1])

        # Calculate the 3D points
        Q1, Q2 = calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = estimate_pose(tp1_l, tp2_l, Q1, Q2)
        
        return transformation_matrix

    start = time.time()

    transf = get_pose()
    New_poseMatrix = np.matmul(Current_poseMatrix, transf)
    New_Position = ((New_poseMatrix[0, 3], New_poseMatrix[2, 3]))

    end = time.time()

    Ellapsed_Time = end-start     

    return New_poseMatrix,New_Position,Ellapsed_Time

def projectOutput(test = 3, directory = 'KITTI_sequence_2'):
    """
    Project output test and real time operation

    Parameters
    ----------
    test (int): 1 - Two image test; 2 - Image test with live plot; 3 - Real-time test (default)
    directory (string): Put in relative path to image directory. Default is 'KITTI_sequence_2'

    Returns
    -------
    None
    """

    #TEST1:
    if test == 1:
        

        data_dir = directory  # Try KITTI_sequence_2
        image_paths = [os.path.join(data_dir + '/image_l', file) for file in sorted(os.listdir(data_dir + '/image_l'))]
        images_l = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        images_l = images_l[0:2]

        image_paths = [os.path.join(data_dir + '/image_r', file) for file in sorted(os.listdir(data_dir + '/image_r'))]
        images_r = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]  
        images_r = images_r[0:2]

        current_poseMatrix = np.array([[1., 0., 0., 0.],[0., 1., 0., 0.],[ 0.,  0.,  1., 0.],[0., 0., 0., 1.]])

        pose_Matrix, position, ellapsed = visualOdometry(images_l,images_r,current_poseMatrix)

        print('Positional Matrix:')
        print(pose_Matrix)
        print('')
        print('New Position:')
        print(position)
        print('')
        print('Ellapsed Time:')
        print(ellapsed)

    # TEST2:
    if test == 2:

        data_dir = directory  # Try KITTI_sequence_2
        image_paths = [os.path.join(data_dir + '/image_l', file) for file in sorted(os.listdir(data_dir + '/image_l'))]
        img_l = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        #images_l = images_l[0:2]

        image_paths = [os.path.join(data_dir + '/image_r', file) for file in sorted(os.listdir(data_dir + '/image_r'))]
        img_r = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]  
        #images_r = images_r[0:2]

        img_l_array=[]
        img_r_array=[]
        for i in range(len(img_l)):

            if i == 0:
                prev_l=img_l[i]
                img_l_array.append(prev_l)
                img_r_array.append(img_r[i])
                current_poseMatrix = np.array([[1., 0., 0., 0.],[0., 1., 0., 0.],[ 0.,  0.,  1., 0.],[0., 0., 0., 1.]])
                estimated_pathx = [0]
                estimated_pathy = [0]

                plt.figure()
                plt.isinteractive()
                plt.title('Estimated Path', fontsize=20)
                plt.xlabel('X axis')
                plt.ylabel('Y axis')
                plt.xlim([-5,50])
                plt.ylim([-5,30])

                plt.plot(estimated_pathx, estimated_pathy,'b')
                plt.show(block=False)
                plt.pause(0.0001)

            else:
                # One frame at a time
                img_l_array.append(img_l[i])
                img_r_array.append(img_r[i])

                # VO call to get new transformation, position,, and time.
                pose_Matrix, position, ellapsed = visualOdometry(img_l_array,img_r_array,current_poseMatrix)
                
                # Updating the new transformation to the current position
                current_poseMatrix = pose_Matrix
                estimated_pathx.append(position[0])
                estimated_pathy.append(position[1])

                # Your previous images becomes the new images. e.g: [1,2], [2, 3], and so on.
                img_l_array=[]
                img_r_array=[]

                img_l_array.append(img_l[i])
                img_r_array.append(img_r[i])
            
                print('Positional Matrix:')
                print(pose_Matrix)
                print('')
                print('New Position:')
                print(position)
                print('')
                print('Ellapsed Time:')
                print(ellapsed)

                plt.plot(estimated_pathx, estimated_pathy,'b')
                plt.show(block=False)
                plt.pause(0.0001)

        plt.show()        

    # TEST3:
    if test == 3:
        cap = cv2.VideoCapture(1)

        i = 0
        img_l_array=[]
        img_r_array=[]

        while cap.isOpened():
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            succes_dual, img_dual = cap.read()
            gray_dual = cv2.cvtColor(img_dual, cv2.COLOR_BGR2GRAY)


            left_right_image = np.split(gray_dual, 2, axis=1)

            cv2.imshow("Left Image", left_right_image[0])
            cv2.imshow("Right Image", left_right_image[1])

            if i == 0:
                img_l_array.append(left_right_image[0])
                img_r_array.append(left_right_image[1])
                current_poseMatrix = np.array([[1., 0., 0., 0.],[0., 1., 0., 0.],[ 0.,  0.,  1., 0.],[0., 0., 0., 1.]])
                estimated_pathx = [0]
                estimated_pathy = [0]

                plt.figure()
                plt.isinteractive()
                plt.title('Estimated Path', fontsize=20)
                plt.xlabel('X axis')
                plt.ylabel('Y axis')
                #plt.xlim([-5,50])
                #plt.ylim([-5,30])

                plt.plot(estimated_pathx, estimated_pathy,'b')
                plt.show(block=False)
                plt.pause(0.0001)

                i+=1

            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break

            else:
                # One frame at a time
                img_l_array.append(left_right_image[0])
                img_r_array.append(left_right_image[1])

                # VO call to get new transformation, position,, and time.
                pose_Matrix, position, ellapsed = visualOdometry(img_l_array,img_r_array,current_poseMatrix)
                
                # Updating the new transformation to the current position
                current_poseMatrix = pose_Matrix
                estimated_pathx.append(position[0])
                estimated_pathy.append(position[1])

                # Your previous images becomes the new images. e.g: [1,2], [2, 3], and so on.
                img_l_array=[]
                img_r_array=[]

                img_l_array.append(left_right_image[0])
                img_r_array.append(left_right_image[1])
            
                print('Positional Matrix:')
                print(pose_Matrix)
                print('')
                print('New Position:')
                print(position)
                print('')
                print('Ellapsed Time:')
                print(ellapsed)

                plt.plot(estimated_pathx, estimated_pathy,'b')
                plt.show(block=False)
                plt.pause(0.0001)
            

        cap.release()

        cv2.destroyAllWindows()
        plt.show()
        
projectOutput()