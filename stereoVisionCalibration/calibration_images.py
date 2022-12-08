import cv2
import numpy as np

cap = cv2.VideoCapture(1)

num = 0

while cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    succes_dual, img_dual = cap.read()

    left_right_image = np.split(img_dual, 2, axis=1)

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('imageL' + str(num) + '.png', left_right_image[0])
        cv2.imwrite('imageR' + str(num) + '.png', left_right_image[1])
        print("images saved!")
        num += 1
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Left Image',left_right_image[0])
    cv2.imshow('Right Image',left_right_image[1])

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
