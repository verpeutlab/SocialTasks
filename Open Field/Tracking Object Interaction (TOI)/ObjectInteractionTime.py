# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:26:34 2023

This code generates the following data for TOI:
    Number of object interactions
    Average object interaction time
    Total time spent in object interaction
"""

# import os
# import numpy as np
# import cv2
# import pandas
# import matplotlib.pyplot as plt
# import csv

# # Load in files generated from createOrderedCenters.py
# order_dir = r'.\new_OrderedCenters'

# mouse_and_ball_files = [f for f in os.listdir(order_dir) if f.endswith('.csv')]

# compare_shift = 5
# fps = 60
# threshold = 34.75 # originally 60
# object_interact_frames_1 = 0
# object_interact_frames_2 = 0

# for mouse_and_ball_file in mouse_and_ball_files:
#     ordered_csv = os.path.join(order_dir, mouse_and_ball_file)
    
#     df_m = pandas.read_csv(ordered_csv)
    
#     # Calculating distance traveled for mouse 1 and mouse 2 (using distance formula)
#     # Calculates distance between mouse 1 and the ball
#     df_m['d1'] = np.sqrt( (df_m['bx']-df_m['x1'])**2 + (df_m['by']-df_m['y1'])**2 )
#     # Calculates distance between mouse 2 and the ball
#     df_m['d2'] = np.sqrt( (df_m['bx']-df_m['x2'])**2 + (df_m['by']-df_m['y2'])**2 )
    
#     # Calculating change in distance for ball (using distance formula)
#     df_m['db'] = np.sqrt( (df_m['bx'].shift(-compare_shift) - df_m['bx'])**2 + (df_m['by'].shift(-compare_shift) - df_m['by'])**2 )
#     df_m['db'].fillna(0, inplace=True)
    
#     # Calculate distance between mouse 1 and mouse 2
#     # df_m['d12'] = np.sqrt( (df_m['x2']-df_m['x1'])**2 + (df_m['y2']-df_m['y1'])**2 )
    
#     # df_m['mx'] = abs(df_m['x1'] - df_m['x2'])
#     # df_m['my'] = abs(df_m['y1'] - df_m['y2'])
    
#     # # Calculate distance between mouse 1/2 and ball
#     # df_m['d12'] = np.sqrt( (df_m['bx']-df_m['mx'])**2 + (df_m['by']-df_m['my'])**2 )
    
#     # Extract indicies of 'db' that are >= 100 so that you can find frames where ball REALLY moved
#     # Goal is to track distance ball traveled in a video
#     df_m['db_diff'] = abs(df_m['db'] - df_m['db'].shift(1))
#     res = []
#     for idx in range(0, len(df_m)):
#         # if df_m['db'][idx] > 600:
#         if df_m['db_diff'][idx] > 640:
#             # I get a reasonable answer here but not sure if logic is right
#             res.append(df_m['db'][idx])
#     print(sum(res))
#     sum_res = sum(res)
#     ball_dist = (sum_res / 1.5) / 1000
#     print(ball_dist)
#     # Note: appears to only decrease slightly when you go up by 100 but a result of zero at 1000

    
#     # In order for object interaction to occur, the distance between a mouse and a ball must be less than the value defined for threshold above
#     df_m['close'] = np.where( ((df_m['d1'] < threshold) | (df_m['d2'] < threshold)), 1, 0 )
    
#     df_m['prev'] = df_m['close'].shift(1)
#     df_m['prev'].fillna(0, inplace=True)
    
#     total_frames = df_m['close'].sum()
#     total_time   = total_frames/fps
#     print('{}: Total frames within {} pixal = {} frames (or {} sec)'.format(mouse_and_ball_file, threshold, total_frames, total_time))
  
#     df_m['in'] = np.where( ((df_m['prev'] == 0) & df_m['close'] == 1), 1, 0 )
#     total_encounters = df_m['in'].sum()
#     print('\tTotal encounters = {} with average time of encounter = {}'.format(total_encounters, total_time/total_encounters))
    
#     # for two mice to be interacting with ball, d1 < value and d2 < value - I think that is the best way to put it
#     df_m['close2'] = np.where( ((df_m['d1'] < 50) & (df_m['d2'] < 50)) , 1, 0 )
#     df_m['prev2'] = df_m['close'].shift(1)
#     df_m['prev2'].fillna(0, inplace=True)
#     total_frames2 = df_m['close2'].sum()
#     total_time2   = total_frames2/fps
#     df_m['in2'] = np.where( ((df_m['prev2'] == 0) & df_m['close2'] == 1), 1, 0 )
#     total_encounters2 = df_m['in2'].sum()
#     print('\tTotal two mice encounters = {} with average time of encounter = {}'.format(total_encounters2, total_time2/total_encounters2))
### This is the good code for ObjectInteractionTime 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

order_dir = r'.\exp_OrderedCenters'
mouse_and_ball_files = [f for f in os.listdir(order_dir) if f.endswith('.csv')]
fps = 60
compare_shift = 5
min_gap = 30  # Minimum gap between encounters in frames
movement_threshold = 5  # Minimum movement to consider as a valid encounter (in pixels)
proximity_threshold = 34.75  # Threshold for close proximity

for mouse_and_ball_file in mouse_and_ball_files:
    ordered_csv = os.path.join(order_dir, mouse_and_ball_file)
    df_m = pd.read_csv(ordered_csv)
    
    # Calculate distances
    df_m['d1'] = np.sqrt((df_m['bx'] - df_m['x1'])**2 + (df_m['by'] - df_m['y1'])**2)
    df_m['d2'] = np.sqrt((df_m['bx'] - df_m['x2'])**2 + (df_m['by'] - df_m['y2'])**2)
    
    # Find dynamic threshold using clustering
    distances = np.concatenate([df_m['d1'].values, df_m['d2'].values])
    distances = distances.reshape(-1, 1)
    
    scaler = StandardScaler()
    distances_scaled = scaler.fit_transform(distances)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(distances_scaled)
    threshold_scaled = kmeans.cluster_centers_.min()
    threshold = scaler.inverse_transform(threshold_scaled.reshape(1, -1))[0][0]
    
    # Detect if close to ball
    df_m['close'] = np.where(((df_m['d1'] < threshold) | (df_m['d2'] < threshold)), 1, 0)
    df_m['prev'] = df_m['close'].shift(1).fillna(0)
    
    # Identify encounters
    df_m['in'] = np.where(((df_m['prev'] == 0) & (df_m['close'] == 1)), 1, 0)
    
    # Calculate total encounters and times
    total_frames = df_m['close'].sum()
    total_time = total_frames / fps
    total_encounters = df_m['in'].sum()
    
    # Filter encounters based on movement
    df_m['movement1'] = np.sqrt((df_m['x1'].shift(-compare_shift) - df_m['x1'])**2 + (df_m['y1'].shift(-compare_shift) - df_m['y1'])**2)
    df_m['movement2'] = np.sqrt((df_m['x2'].shift(-compare_shift) - df_m['x2'])**2 + (df_m['y2'].shift(-compare_shift) - df_m['y2'])**2)
    
    df_m['valid_encounter'] = np.where((df_m['movement1'] > movement_threshold) | (df_m['movement2'] > movement_threshold), 1, 0)
    df_m['filtered_encounter'] = df_m['in'] * df_m['valid_encounter']
    
    filtered_total_time = df_m[df_m['filtered_encounter'] == 1]['close'].sum() / fps
    average_time = filtered_total_time / total_encounters if total_encounters > 0 else 0
    
    # For two mice encounters
    df_m['close2'] = np.where(((df_m['d1'] < 50) & (df_m['d2'] < 50)), 1, 0)
    df_m['prev2'] = df_m['close2'].shift(1).fillna(0)
    df_m['in2'] = np.where(((df_m['prev2'] == 0) & (df_m['close2'] == 1)), 1, 0)
    
    total_frames2 = df_m['close2'].sum()
    total_time2 = total_frames2 / fps
    total_encounters2 = df_m['in2'].sum()
    average_time2 = total_time2 / total_encounters2 if total_encounters2 > 0 else 0
    
    # Calculate one mouse encounters
    one_mouse_encounters = total_encounters - total_encounters2
    one_mouse_average_time = filtered_total_time / one_mouse_encounters if one_mouse_encounters > 0 else 0
    
    print('{}: Total encounters = {}'.format(mouse_and_ball_file, total_encounters))
    print('\tOne mouse encounters = {} with average time = {:.4f} sec'.format(one_mouse_encounters, one_mouse_average_time))
    print('\tTotal time spent in filtered encounters = {:.4f} sec'.format(filtered_total_time))
    print('\tTotal two mice encounters = {} with average time = {:.4f} sec'.format(total_encounters2, average_time2))

    # Save output if needed
    # df_m.to_csv(some_output_path, index=False)
