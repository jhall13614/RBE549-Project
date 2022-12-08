import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import time
from Historical.Historical.RBE549_Stereo_VO__Project_Code_JHall_11282022 import visualOdometry
import matplotlib.pyplot as plt

#from lib.visualization.plotting import visualize_paths
#from lib.visualization.video import play_trip

#from tqdm import tqdm
"""
def main():

    # Setting and initializing parameters
    skip_frames = 2 # we can play within to see what would look best and is less time consuming
    data_directory = ""

    # If I am not using this VO then we need to find a new way to call this
    vo = VisualOdometry(data_directory,stereo,feature,optical)

    # Create empty list for the ground true, estimated path, and camera poses
    # play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip

    # expected_path = []
    estimated_path = []
    camera_poses = []

    # Home base with positions as zero in x, y, z
    starting_position = np.array([0, 0, 0, 1])

    # We now need to initilize the start pose - translation and rotation
    initial_pose = np.ones((3,4)) # the iinitial pose would be matrix of 3 x 4 composed of 1s
    initial_rotation = np.identity(3) # Identity matrix - meaning no rotation
    initial_translation = np.zeros((3,1)) # x, y, and z all being zero
    initial_transformation = np.concatenate((initial_rotation, initial_translation), axis = 1)
    current_location = initial_transformation

    # This is simply Transformation_matrix = np.array([[1, 0, 0, dX],
    #                                                   [0, 1, 0, dY],
    #                                                   [0, 0, 1, dZ],
    #                                                   [0, 0, 0, 1]])

    # To capture a video, we need to create an object for the VideoCapture class.
    # It accepts either a device index or the name of a video file.
    camera1 = cv2.VideoCapture(0) # Means first camera or webcam
    camera2 = cv2.VideoCapture(1) # Means second camera or webcam
    # Resolution for camera is optional. For now, I will leave it as default

    # Initialize frame counter
    counter = 0

    # Initialize frames old and new
    frameByFrame = 0
    previous_left_frame = None
    previous_right_frame = None
    next_left_frame = None
    next_right_frame = None

    # We then use a while loop to read the video frame by frame and pass it to the cv2.imshow() function 
    # inside the while loop. If a frame is read incorrectly then the loop will break.
    while(camera1 and camera2 == True):
        # The cap.read() returns the boolean value(True/False).It will return True, if the frame is read correctly.
        ret1, next_left_frame = camera1.read()
        ret2, next_right_frame = camera2.read()
        # Loading video in grayscale mode
        left_frame_gray = cv2.cvtColor(next_left_frame, cv2.COLOR_BGR2GRAY)
        right_frame_gray = cv2.cvtColor(next_right_frame, cv2.COLOR_BGR2GRAY)
        # Counting the number of frames while the cameras are on
        counter = counter + 1

        # Now we can use OV.getPose() to get the actual transformation and estimated path
        while(frameByFrame and ret1 != False):

            need to continue here for case of current transformation + adding them to the
            list of estimated values so that we can plot them in 3D as we go
            transformation = vo.getpose(previous_left_frame, previous_right_frame,
            left_frame_gray, right_frame_gray)

            # It multiples each transformation to the current location per iteration
            current_location = current_location * transformation
            # New camera position based on the current location
            camera_position = np.concatenate(current_location, starting_position)
            # Previously initializes as empty lists
            camera_poses.append(camera_position)
            estimated_path.append(current_location[0,3], current_location[2,3])

            pose_x, pose_y = current_location[0,3], current_location[2,3]

            # If the process frame 
            if frameByFrame and ret1 is True:
                break

        previous_left_frame = left_frame_gray
        previous_right_frame = right_frame_gray

        frameByFrame = True

        # displays video on a window
        cv2.imshow('Left Camera', next_left_frame)
        cv2.imshow('Right Camera', next_right_frame)
        # Until we press the q key or 1 key, it will keep opening a web camera and capture the Video.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()

    Real time 3D Plotting will go below 

    image_window = np.array([1280, 720])

    plt.figure()

    # Takes every 2 transformation from the camera pose
    estimated_path = np.array(estimated_path[::2])

    plt.plot(estimated_path[:,0], estimated_path[:,1])
    plt.title('Estimated Path', fontsize=20)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()


if __name__ == "__main__":
    main()
"""

# Array input call
def images_array(i):

     # Get the i-1'th image and i'th image
    old_l_img, new_l_img = images_l[i:i + 1]
    images_l = [old_l_img, new_l_img]

    old_r_img, new_r_img = images_r[i:i + 1]
    images_r = [old_r_img, new_r_img]

    return images_l, images_r

# Defining camera to capture images (2)
cap = cv2.VideoCapture(2)

# Initilize estimated path 
estimated_path = []

i=0
while(True):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Need to divide camera into two sections
    camera_l = frame[:,0:320,:]
    camera_r = frame[:,320:640,:]

    if i:
        images_l = None
        images_r = None
        start_poseMatrix = np.array([[1., 0., 0., 0.],[0., 1., 0., 0.],[ 0.,  0.,  1., 0.],[0., 0., 0., 1.]])
        current_poseMatrix = start_poseMatrix
        i+=1
    else:
        # Left_Images ([Previous Image, Current Image]): Current and previous images from left camera
        # Right_Images ([Previous Image, Current Image]): Current and previous images from left camera
        # Current_poseMatrix ([4x4]): Current positional matrix as a 4x4 array

        # already input in array form from fucntion images_l = [old_l_img, new_l_img]
        lf_img_l, lf_img_r = images_array(camera_l) 
        rf_img_l, rf_img_r = images_array(camera_r)

        left_camera = [lf_img_l, lf_img_r]
        right_camera = [rf_img_l, rf_img_r]

        current_poseMatrix = current_poseMatrix + 1

    pose_Matrix, position, ellapsed = visualOdometry(left_camera,right_camera,current_poseMatrix)

    # Multiply new position to the new pose
    current_poseMatrix = pose_Matrix

    estimated_path.append(position)

    #cv2.imshow('frame', frame)
    cv2.imshow('Left frame = {}'.format(camera_l.shape), camera_l)
    cv2.imshow('Right frame = {}'.format(camera_r.shape), camera_r)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

""" Real time 3D Plotting will go below """
"""
image_window = np.array([1280, 720])

plt.figure()

# Takes every 2 transformation from the camera pose
estimated_path = np.array(estimated_path[::2])

plt.plot(estimated_path[:,0], estimated_path[:,1])
plt.title('Estimated Path', fontsize=20)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()"""