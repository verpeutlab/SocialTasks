# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:38:34 2023

This is the first script in the TOI code.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import pickle

ball_width = 40
search_box = 300
method     = cv2.TM_SQDIFF_NORMED
video_dir  = r'.\new_videos'
track_dir  = r'.\new_ball_location'
ball_dir   = r'.\new_marked_video'
rescale_dir = r'.\new_rescale_param'

# Save all video files in the same folder. One line for each video that will be analyzed.
# Set dimensions of video. This crops the videos a little bit, as it takes a while to run this code (about a day for 16 videos)
known_files = dict()
# known_files[ 'OF_C349NSaline-10252023092328.avi' ] = [ 100, 840, 250, 1000, 440, 540, 150 ]
# known_files[ 'OF_C346N-10062023120112.avi' ] = [ 100, 840, 250, 1000, 475, 650, 150 ]
# known_files[ 'OF_C346NSaline.avi'  ] = [ 100, 840, 250, 1000, 340, 370, 150 ]
# known_files[ 'OF_C347N.avi'  ] = [ 100, 840, 250, 1000, 380, 405, 150 ]
# known_files[ 'OF_C347NSaline.avi'  ] = [ 100, 840, 250, 1000, 490, 860, 100 ]
# known_files[ 'OF_C348N.avi'  ] = [ 100, 840, 250, 1000, 440, 775, 100 ]
known_files[ 'OF_C257R-12062022102746.avi'  ] = [ 100, 840, 250, 1000, 440, 710, 100 ]
# known_files[ 'OF_C349N.avi'  ] = [ 100, 840, 250, 1000, 430, 655, 100 ]
# known_files[ 'OF_C289L-02092023101635.avi'  ] = [ 100, 840, 250, 1000, 360, 635, 100 ]
# known_files[ 'OF_C292L-03012023084310.avi'  ] = [ 100, 840, 250, 1000, 520, 680, 100 ]
# known_files[ 'OF_C296LLR-03012023100105.avi'] = [ 100, 840, 250, 1000, 505, 655, 100 ]
# known_files[ 'OF_C296RRL-03012023092206.avi'] = [ 100, 840, 250, 1000, 490, 640, 100 ]
# known_files[ 'OF_C302L-03162023084114.avi'  ] = [ 100, 840, 250, 1000, 440, 510, 100 ]
# known_files[ 'OF_C303L-03162023091949.avi'  ] = [ 100, 840, 250, 1000, 340, 415, 100 ]
# known_files[ 'OF_C308L-03232023083758.avi'  ] = [ 100, 840, 250, 1000, 475, 620, 100 ]
# known_files[ 'OF_C313L-04202023083250.avi'  ] = [ 100, 840, 250, 1000, 520, 645, 100 ]

# Function called later in script. Max pixel is the pixel with the maximum value in a frame.
# This is what makes the mice appear white and the background to appear black.
def RescaleImage( input_image, max_pixel, rescale_value ):
    inverse_image  = max_pixel - np.asarray(input_image, dtype='uint8')
    rescaled_image = np.asarray( np.multiply( rescale_value, np.asarray( inverse_image, dtype=float ) ).round(), dtype='uint8' )
    
    return rescaled_image

def GetReference( full_avi, frame_data ):
    
    cap    = cv2.VideoCapture( full_avi )
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    min_pixel = 255 * np.ones(  [frame_data[1]-frame_data[0], frame_data[3]-frame_data[2]], dtype='uint8' )
    max_pixel =       np.zeros( [frame_data[1]-frame_data[0], frame_data[3]-frame_data[2]], dtype='uint8' )
    
    for cnt in range(0, length):
        if cnt%100 == 0:
            print( 'Rescaling frame {} of {}'.format( cnt, length ) )
            
        ret, frame = cap.read()
        grayImage  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        sub_image = grayImage[frame_data[0]:frame_data[1], frame_data[2]:frame_data[3]]
        np_image  = np.asarray(sub_image, dtype='uint8')
        
        min_pixel = np.minimum( min_pixel, np_image )
        max_pixel = np.maximum( max_pixel, np_image )
        
        if cnt == frame_data[6]:
            full_ball = sub_image
            
    cap.release()
    
    # max_pixel_orig = max_pixel.copy()
    # for i in range(0, frame_data[1]-frame_data[0]):
    #     xi = list( range(max(0, i-5), min(frame_data[1]-frame_data[0]-1, i+5)) )
    #     for j in range(0, frame_data[3]-frame_data[2]):
    #         yi = list( range(max(0, j-5), min(frame_data[3]-frame_data[2]-1, j+5)) )
            
    #         max_pixel[i, j] = min(map(min, max_pixel_orig[np.ix_(xi,yi)]))
    
    rescale_value = np.divide( 250 * np.ones([frame_data[1]-frame_data[0], frame_data[3]-frame_data[2]]),
                               max_pixel )
    
    for xi in range(0, len(max_pixel)):
        for yi in range(0, len(max_pixel[0])):
            if max_pixel[xi][yi] - min_pixel[xi][yi] < 35:
                rescale_value[xi][yi] = 0
    
    full_image_rescale = RescaleImage( full_ball, max_pixel, rescale_value )
    
    ball_image = full_image_rescale[ frame_data[4]-frame_data[0]-ball_width:frame_data[4]-frame_data[0]+ball_width,
                                     frame_data[5]-frame_data[2]-ball_width:frame_data[5]-frame_data[2]+ball_width]
    
    return ball_image, max_pixel, rescale_value
## This begins the python code that will be ran, not the functions.
if __name__ == '__main__':
    # avi_files lists all videos in video_dir
    # pkl_files lists all pkl viles in rescale_dir. Handy if you stopped midway through and are starting again. 
    # If a pkl file has already been created, it will not be replaced.
    # This script will create a pkl file for each video in avi_files
    avi_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]
    pkl_files = [f for f in os.listdir(rescale_dir) if f.endswith('.pkl')]
    
    for avi_file in avi_files:
    
        full_avi = os.path.join(video_dir, avi_file)
        # Crop video frame-by-frame so that only pixels inside open field are analyzed
        frame_data = known_files[ avi_file ]
        
        # Creating pkl output file
        tmp = list(avi_file)
        tmp[-3] = 'p'
        tmp[-2] = 'k'
        tmp[-1] = 'l'
        pkl_file = "".join(tmp)
        if pkl_file in pkl_files:
            continue
        
        full_pkl = os.path.join(rescale_dir, pkl_file)
       
        # Call GetReference function, which incorporates RescaleImage function. Both functions were created in this script
        # In the GetReference function, you are going frame by frame and updating the max and min for each pixel.
        # These results are saved into the pkl file.
        ball_image, max_pixel, rescale_value = GetReference( full_avi, frame_data )
        
        with open(full_pkl, 'wb') as f:
            pickle.dump([ball_image, max_pixel, rescale_value], f)
            
        
        