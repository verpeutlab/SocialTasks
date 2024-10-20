# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 18:27:22 2023

 This is the second script ran in TOI code.
"""
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.cluster import KMeans

ball_width = 40
search_box = 300
method     = cv2.TM_SQDIFF_NORMED
video_dir  = r'.\new_videos'
track_dir  = r'.\new_ball_location'
ball_dir   = r'.\new_marked_video'
rescale_dir = r'.\new_rescale_param'

known_files = dict()
# known_files[ 'OF_C346N-10062023120112.avi' ] = [ 100, 840, 250, 1000, 475, 650, 150 ]
# known_files[ 'OF_C346NSaline.avi'  ] = [ 100, 840, 250, 1000, 340, 370, 150 ]
# known_files[ 'OF_C347N.avi'  ] = [ 100, 840, 250, 1000, 380, 405, 150 ]
# known_files[ 'OF_C347NSaline.avi'  ] = [ 100, 840, 250, 1000, 490, 860, 100 ]
# known_files[ 'OF_C348N.avi'  ] = [ 100, 840, 250, 1000, 440, 775, 100 ]
known_files[ 'OF_C257R-12062022102746.avi'  ] = [ 100, 840, 250, 1000, 440, 710, 100 ]
#known_files[ 'OF_C349N.avi'  ] = [ 100, 840, 250, 1000, 430, 655, 100 ]
#known_files[ 'OF_C349NSaline-10252023092328.avi' ] = [ 100, 840, 250, 1000, 440, 540, 150 ]
# known_files[ 'OF_C249LR-10202022091104.avi' ] = [ 100, 840, 250, 1000, 440, 540, 150 ]
# known_files[ 'OF_C249RR-10202022095215.avi' ] = [ 100, 840, 250, 1000, 475, 650, 150 ]
# known_files[ 'OF_C255L-12012022105404.avi'  ] = [ 100, 840, 250, 1000, 340, 370, 150 ]
# known_files[ 'OF_C257L-12062022110059.avi'  ] = [ 100, 840, 250, 1000, 380, 405, 150 ]
# known_files[ 'OF_C257R-12062022102746.avi'  ] = [ 100, 840, 250, 1000, 490, 860, 100 ]
# known_files[ 'OF_C262L-12162022093107.avi'  ] = [ 100, 840, 250, 1000, 440, 775, 100 ]
# known_files[ 'OF_C262R-12162022100712.avi'  ] = [ 100, 840, 250, 1000, 440, 710, 100 ]
# known_files[ 'OF_C274L-01112023085951.avi'  ] = [ 100, 840, 250, 1000, 430, 655, 100 ]
# known_files[ 'OF_C289L-02092023101635.avi'  ] = [ 100, 840, 250, 1000, 360, 635, 100 ]
# known_files[ 'OF_C292L-03012023084310.avi'  ] = [ 100, 840, 250, 1000, 520, 680, 100 ]
# known_files[ 'OF_C296LLR-03012023100105.avi'] = [ 100, 840, 250, 1000, 505, 655, 100 ]
# known_files[ 'OF_C296RRL-03012023092206.avi'] = [ 100, 840, 250, 1000, 490, 640, 100 ]
# known_files[ 'OF_C302L-03162023084114.avi'  ] = [ 100, 840, 250, 1000, 440, 510, 100 ]
# known_files[ 'OF_C303L-03162023091949.avi'  ] = [ 100, 840, 250, 1000, 340, 415, 100 ]
# known_files[ 'OF_C308L-03232023083758.avi'  ] = [ 100, 840, 250, 1000, 475, 620, 100 ]
# known_files[ 'OF_C313L-04202023083250.avi'  ] = [ 100, 840, 250, 1000, 520, 645, 100 ]

def PlotFrame( avi_file ):
    print( 'AVI file: {}'.format(avi_file) )
    
    full_avi = os.path.join(video_dir, avi_file)
    cap    = cv2.VideoCapture( full_avi )
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    for cnt in range(0, 100):
        ret, frame = cap.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cap.release()
    
    plt.figure()
    plt.imshow(grayImage, vmin=0, vmax=255)
    
def RescaleImage( input_image, max_pixel, rescale_value ):
    inverse_image  = max_pixel - np.asarray(input_image, dtype='uint8')
    rescaled_image = np.asarray( np.multiply( rescale_value, np.asarray( inverse_image, dtype=float ) ).round(), dtype='uint8' )
    
    return rescaled_image
    

def GetBallLoc( frame, ball_image ):
    result = cv2.matchTemplate(ball_image, frame, method)
    # We want the minimum squared difference
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)
    
    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = mnLoc
    
    return MPx,MPy,mn
# Main Python code begins here    
if __name__ == '__main__':
    
    h = 10
    p = 150
    h2 = 2*h+1
    t = h2**2 * p * .75
    
    avi_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]
    csv_files = [f for f in os.listdir(track_dir) if f.endswith('.csv')]
    ball_files = [f for f in os.listdir(ball_dir ) if f.endswith('.avi')]
    
    for avi_file in avi_files:
        # PlotFrame( avi_file )
        # break
        if not avi_file in known_files:
            PlotFrame( avi_file )
            break
    
        full_avi = os.path.join(video_dir, avi_file)
        frame_data = known_files[ avi_file ]
        
        tmp = list(avi_file)
        tmp[-3] = 'c'
        tmp[-2] = 's'
        tmp[-1] = 'v'
        csv_file = "".join(tmp)
        full_csv = os.path.join(track_dir, csv_file)
        
        tmp = list(avi_file)
        tmp[-3] = 'p'
        tmp[-2] = 'k'
        tmp[-1] = 'l'
        pkl_file = "".join(tmp)
        full_pkl = os.path.join(rescale_dir, pkl_file)
        
        tmp[-4] = '_'
        tmp[-3] = 'c'
        tmp[-2] = 'a'
        tmp[-1] = 'l'
        tmp.append('l')
        tmp.append('.')
        tmp.append('a')
        tmp.append('v')
        tmp.append('i')
        ball_file = "".join(tmp)
        full_ball = os.path.join(ball_dir , ball_file)
        
        if not csv_file in csv_files:
            print( "Tracking {}".format( avi_file ) )
            # Import pkl files from rescaleParam.py
            with open(full_pkl, 'rb') as pf:
                ball_image, max_pixel, rescale_value = pickle.load(pf)
            
            cap    = cv2.VideoCapture( full_avi )
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            
            MPx = 0
            MPy = 0
            
            min_x = frame_data[0]
            min_y = frame_data[2]
            max_x = frame_data[1]
            max_y = frame_data[3]
            
            f = open(full_csv, "w")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = ( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ))
            out_file = cv2.VideoWriter( full_ball, fourcc, fps, frame_size )
            
            for cnt in range(0, length):
                if cnt%100 == 0:
                    print( '{} : Frame {} of {}'.format( avi_file, cnt, length ) )
                ret, frame = cap.read()
                
                grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sub_image = grayImage[min_x:max_x,min_y:max_y]
                # Run RescaleImage funcrtion defined in resecaleParam.py
                # This function makes sure all pixels have the same distribution from the min value to the max value.
                image_rescale = RescaleImage( sub_image, max_pixel, rescale_value )
                
                # Final noise check. Checks to see if each pixel that meets the threshold is surrounded by other pixels that meet the threshold. 
                # If a pixel is not surrounded by other pixels, then it is removed. 
                # This process causes the mouse's tail to be removed.
                image_rescale[image_rescale<p] = 0
                
                sum_image = np.asarray(image_rescale, dtype=int )
                
                sum_image = np.cumsum(sum_image,axis=0)
                sum_image[(h2+1):,:] = sum_image[(h2+1):,:] - sum_image[:-(h2+1),:]
                sum_image[:-(h+1),:] = sum_image[(h+1):,:]
                
                sum_image = np.cumsum(sum_image,axis=1)
                sum_image[:,(h2+1):] = sum_image[:,(h2+1):] - sum_image[:,:-(h2+1)]
                sum_image[:,:-(h+1)] = sum_image[:,(h+1):]
                
                image_rescale[sum_image<t] = 0
                scaled_features = np.argwhere(image_rescale>0)
                
                # All ponits that meet the threshold are fed into k-means so that three clusters (2 mice and 1 ball) can be created
                kmeans = KMeans( init="random", n_clusters=3, n_init=10,  max_iter=300, random_state=42 )
                kmeans.fit(scaled_features)
                centers = kmeans.cluster_centers_
                
                # for ii in np.argwhere(image_rescale<180):
                #     image_rescale[ii[0], ii[1]] = 0
                    
                # MPx, MPy, min_value = GetBallLoc( image_rescale, ball_image )
                # # print('{} + {} + {}, {} + {} + {}'.format(MPy , min_x , ball_width, MPx , min_y , ball_width))
                # MPy = MPy + min_x + ball_width
                # MPx = MPx + min_y + ball_width
                
                
                
                # x = MPy
                # y = MPx
                
                frame[:][:][:] = 0
                frame[min_x:max_x, min_y:max_y, 0] = image_rescale
                frame[min_x:max_x, min_y:max_y, 1] = image_rescale
                frame[min_x:max_x, min_y:max_y, 2] = image_rescale
                
                f.write("{}".format( cnt ))
                for center in centers:
                    x = int(np.round(center[0])) + min_x
                    y = int(np.round(center[1])) + min_y
                    
                    f.write(",{},{}\n".format( y, x ))
                    
                    #print('{} + {}, {} + {}'.format(x , min_x , y , min_y))
                
                    for xx in range(-5,6):
                        for yy in range(-5,6):
                            if (xx == yy) or (xx + yy == 0):
                                frame[x+xx][y+yy] = [0, 0, 0]
                            else:
                                frame[x+xx][y+yy] = [0, 255, 0]
                
                f.write("\n")
                out_file.write(frame)
                
            f.close()
            cap.release()
            out_file.release()
        
        