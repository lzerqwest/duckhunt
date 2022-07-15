from pickle import NONE
import time
from cv2 import ROTATE_90_CLOCKWISE, threshold
import numpy as np 
import cv2

previous_frame = None

#to store frames
frames = []
threshframes = []

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
def GetLocation(move_type, env, current_frame):
    time.sleep(1) #artificial one second processing time

    global previous_frame

    #for median and gaussian blur
    median_size = 9
    gaussian_size = 5

    #erosion and dilation
    erode_size = 7
    dilate_size = 30

    #drawing contours
    threshold = 50
    contour_min = 1000.0
    contour_max = 2000.0
    
    #Use relative coordinates to the current position of the "gun", defined as an integer below
    if move_type == "relative":
        """
        North = 0
        North-East = 1
        East = 2
        South-East = 3
        South = 4
        South-West = 5
        West = 6
        North-West = 7
        NOOP = 8
        """
        coordinate = env.action_space.sample() 
    #Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (1024, 798) 
        
        """
        cv2.waitKey(100)

        #a copy of the frame needs to be saved
        current_copy = current_frame

        #for processing
        test_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

        #median blur
        test_frame = cv2.medianBlur(src=test_frame, ksize=median_size)
        #gaussian blur
        test_frame = cv2.GaussianBlur(src=test_frame, ksize=(gaussian_size, gaussian_size), sigmaX=0)
        
        # for the first run
        if previous_frame is None:
            previous_frame = test_frame

        # find the absolute difference between previous frame and the current one
        diff_frame = cv2.absdiff(previous_frame, test_frame)
        previous_frame = test_frame

        # kernel for erosion and dilation
        erode_kernel = np.ones((erode_size, erode_size))
        dilate_kernel = np.ones((dilate_size, dilate_size))

        # apply a thresholding to diff_frame to remove small differences (movements)
        threshold_frame = cv2.threshold(src=diff_frame, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        
        # erode result to make the resultant 'blobs' of movement smaller
        threshold_frame = cv2.erode(src=threshold_frame, kernel=erode_kernel)
        threshold_frame = cv2.dilate(src=threshold_frame, kernel=dilate_kernel)

        width = int(threshold_frame.shape[1] * 60 / 100)
        height = int(threshold_frame.shape[0] * 60 / 100)
        dim = (width, height)
  
        # resize image
        threshold_frame = cv2.resize(threshold_frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("threshold",threshold_frame)

        # create contours from absdiff
        contours, _ = cv2.findContours(image=threshold_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1)
        for contour in contours:
            # contour cant be too large or too small
            if (cv2.contourArea(contour) > contour_min and cv2.contourArea(contour) < contour_max):
                (x,y), _ = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                cv2.circle(current_copy, center, radius=20, color=(255, 0, 0), thickness=2)
                frames.append(center)
        
        # draw contour
        cv2.drawContours(image=current_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
        current_copy = np.swapaxes(current_copy,0,1) 
        width = int(current_copy.shape[1] * 60 / 100)
        height = int(current_copy.shape[0] * 60 / 100)
        dim = (width, height)
        current_copy = cv2.resize(current_copy, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("diff",current_copy)

        coordinate = env.action_space_abs.sample()

        #frame = cv2.rotate(current_frame,ROTATE_90_CLOCKWISE)

        #img = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        #img = np.swapaxes(img,0,1) 

        #previmg = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
        #previmg = np.swapaxes(previmg,0,1)
        #diff = cv2.absdiff(previmg, img)
        #thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        #current_copy = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
      

        #print(diff)

        #frames.append(img)
        # Calculate the median along the time axis
        #medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)  
        # Display median frame
        #cv2.imshow('frame', threshholded_frame)

        #edges = cv2.Canny(img,100,200,5)
        #cv2.imshow('okay',edges)

        #cv2.imshow('hmm', previous_frame)

        #corners = cv2.goodFeaturesToTrack(img,25,0.01,10)
        #cv2.imshow('hmm',corners)
    
    return [{'coordinate' : coordinate, 'move_type' : move_type}]

