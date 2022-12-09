import cv2
import numpy as np
import matplotlib.pyplot as plt

# define a video capture object
vid = cv2.VideoCapture(2)

estimated_path = []
  
while(vid.isOpened()):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    camera_l = frame[:,0:320,:]
    camera_r = frame[:,320:640,:]
  
    # Display the resulting frame
    cv2.imshow('Left frame', camera_l)
    cv2.imshow('Right frame', camera_r)

    #estimated_path.append(camera_l[0:2])
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

""" Real time 3D Plotting will go below """

"""image_window = np.array([1280, 720])

plt.figure()

# Takes every 2 transformation from the camera pose
estimated_path = np.array(estimated_path[::2])
print(estimated_path.shape)

plt.plot(estimated_path[:,0], estimated_path[:,1])
plt.title('Estimated Path', fontsize=20)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()"""