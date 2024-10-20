# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:22:31 2024

@author: markd
"""

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

# order_dir = r'.\OrderedCenters1'

# mouse_and_ball_files = [f for f in os.listdir(order_dir) if f.endswith('.csv')]

# fps = 60
# compare_shift = 5

# for mouse_and_ball_file in mouse_and_ball_files:
#     ordered_csv = os.path.join(order_dir, mouse_and_ball_file)
#     df_m = pd.read_csv(ordered_csv)
    
#     # Calculate distances
#     df_m['d1'] = np.sqrt((df_m['bx'] - df_m['x1'])**2 + (df_m['by'] - df_m['y1'])**2)
#     df_m['d2'] = np.sqrt((df_m['bx'] - df_m['x2'])**2 + (df_m['by'] - df_m['y2'])**2)
    
#     # Find dynamic threshold using clustering or other methods
#     distances = np.concatenate([df_m['d1'].values, df_m['d2'].values])
#     distances = distances.reshape(-1, 1)  # Reshape to 2D array
    
#     scaler = StandardScaler()
#     distances_scaled = scaler.fit_transform(distances)
    
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(distances_scaled)
#     threshold_scaled = kmeans.cluster_centers_.min()
#     threshold = scaler.inverse_transform(threshold_scaled.reshape(1, -1))[0][0]
    
#     df_m['close'] = np.where(((df_m['d1'] < threshold) | (df_m['d2'] < threshold)), 1, 0)
#     df_m['prev'] = df_m['close'].shift(1).fillna(0)
    
#     total_frames = df_m['close'].sum()
#     total_time = total_frames / fps
#     print('{}: Total frames within threshold = {} frames (or {} sec)'.format(mouse_and_ball_file, total_frames, total_time))
    
#     df_m['in'] = np.where(((df_m['prev'] == 0) & (df_m['close'] == 1)), 1, 0)
#     total_encounters = df_m['in'].sum()
#     print('\tTotal encounters = {} with average time of encounter = {}'.format(total_encounters, total_time / total_encounters if total_encounters > 0 else 'N/A'))
    
#     # For two mice encounters
#     df_m['close2'] = np.where(((df_m['d1'] < 50) & (df_m['d2'] < 50)), 1, 0)
#     df_m['prev2'] = df_m['close2'].shift(1).fillna(0)
    
#     total_frames2 = df_m['close2'].sum()
#     total_time2 = total_frames2 / fps
#     df_m['in2'] = np.where(((df_m['prev2'] == 0) & (df_m['close2'] == 1)), 1, 0)
#     total_encounters2 = df_m['in2'].sum()
#     print('\tTotal two mice encounters = {} with average time of encounter = {}'.format(total_encounters2, total_time2 / total_encounters2 if total_encounters2 > 0 else 'N/A'))
    
#     # Add time bin analysis here
#     video_duration_seconds = 1200  # 20 minutes = 1200 seconds
#     time_bin_size_seconds = 60     # 1 minute bins

#     # Create time bins
#     bins = np.arange(0, video_duration_seconds + time_bin_size_seconds, time_bin_size_seconds)
#     bin_labels = [f'{i}-{i + time_bin_size_seconds}' for i in bins[:-1]]

#     # Initialize result dictionaries
#     two_mice_encounters_per_minute = {label: 0 for label in bin_labels}
#     one_mouse_encounters_per_minute = {label: 0 for label in bin_labels}

#     # Function to categorize encounters into time bins
#     def categorize_encounters(df):
#         for index, row in df.iterrows():
#             start_time = index / fps  # Convert frame index to time in seconds
#             bin_index = np.digitize(start_time, bins) - 1
#             if bin_index >= len(bin_labels):
#                 bin_index = len(bin_labels) - 1  # Handle index out of range for the last bin
#             if row['in2'] == 1:
#                 two_mice_encounters_per_minute[bin_labels[bin_index]] += 1
#             if row['in'] == 1 and row['in2'] == 0:
#                 one_mouse_encounters_per_minute[bin_labels[bin_index]] += 1

#     # Categorize encounters
#     categorize_encounters(df_m)

#     print("\nTwo Mouse Encounters Per Minute:")
#     for label, count in two_mice_encounters_per_minute.items():
#         print(f'{label}: {count}')

#     print("\nOne Mouse Encounters Per Minute:")
#     for label, count in one_mouse_encounters_per_minute.items():
#         print(f'{label}: {count}')

#     total_one_mouse_encounters = total_encounters - total_encounters2
#     print(f"\nTotal One Mouse Encounters: {total_one_mouse_encounters}")

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

order_dir = r'.\exp_OrderedCenters'

mouse_and_ball_files = [f for f in os.listdir(order_dir) if f.endswith('.csv')]

fps = 60
compare_shift = 5

for mouse_and_ball_file in mouse_and_ball_files:
    ordered_csv = os.path.join(order_dir, mouse_and_ball_file)
    df_m = pd.read_csv(ordered_csv)
    
    # Calculate distances
    df_m['d1'] = np.sqrt((df_m['bx'] - df_m['x1'])**2 + (df_m['by'] - df_m['y1'])**2)
    df_m['d2'] = np.sqrt((df_m['bx'] - df_m['x2'])**2 + (df_m['by'] - df_m['y2'])**2)
    
    # Find dynamic threshold using clustering or other methods
    distances = np.concatenate([df_m['d1'].values, df_m['d2'].values])
    distances = distances.reshape(-1, 1)  # Reshape to 2D array
    
    scaler = StandardScaler()
    distances_scaled = scaler.fit_transform(distances)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(distances_scaled)
    threshold_scaled = kmeans.cluster_centers_.min()
    threshold = scaler.inverse_transform(threshold_scaled.reshape(1, -1))[0][0]
    
    df_m['close'] = np.where(((df_m['d1'] < threshold) | (df_m['d2'] < threshold)), 1, 0)
    df_m['prev'] = df_m['close'].shift(1).fillna(0)
    
    total_frames = df_m['close'].sum()
    total_time = total_frames / fps
    print('{}: Total frames within threshold = {} frames (or {} sec)'.format(mouse_and_ball_file, total_frames, total_time))
    
    df_m['in'] = np.where(((df_m['prev'] == 0) & (df_m['close'] == 1)), 1, 0)
    total_encounters = df_m['in'].sum()
    print('\tTotal encounters = {} with average time of encounter = {}'.format(total_encounters, total_time / total_encounters if total_encounters > 0 else 'N/A'))
    
    # For two mice encounters
    df_m['close2'] = np.where(((df_m['d1'] < 50) & (df_m['d2'] < 50)), 1, 0)
    df_m['prev2'] = df_m['close2'].shift(1).fillna(0)
    
    total_frames2 = df_m['close2'].sum()
    total_time2 = total_frames2 / fps
    df_m['in2'] = np.where(((df_m['prev2'] == 0) & (df_m['close2'] == 1)), 1, 0)
    total_encounters2 = df_m['in2'].sum()
    print('\tTotal two mice encounters = {} with average time of encounter = {}'.format(total_encounters2, total_time2 / total_encounters2 if total_encounters2 > 0 else 'N/A'))
    
    # Add time bin analysis here
    video_duration_seconds = 1200  # 20 minutes = 1200 seconds
    time_bin_size_seconds = 60     # 1 minute bins

    # Create time bins
    bins = np.arange(0, video_duration_seconds + time_bin_size_seconds, time_bin_size_seconds)
    bin_labels = [f'{i}-{i + time_bin_size_seconds}' for i in bins[:-1]]

    # Initialize result dictionaries
    two_mice_encounters_per_minute = {label: 0 for label in bin_labels}
    one_mouse_encounters_per_minute = {label: 0 for label in bin_labels}

    # Function to categorize encounters into time bins
    def categorize_encounters(df):
        for index, row in df.iterrows():
            start_time = index / fps  # Convert frame index to time in seconds
            bin_index = np.digitize(start_time, bins) - 1
            if bin_index >= len(bin_labels):
                bin_index = len(bin_labels) - 1  # Handle index out of range for the last bin
            if row['in2'] == 1:
                two_mice_encounters_per_minute[bin_labels[bin_index]] += 1
            elif row['in'] == 1 and row['in2'] == 0:
                one_mouse_encounters_per_minute[bin_labels[bin_index]] += 1

    # Categorize encounters
    categorize_encounters(df_m)

    print("\nTwo Mouse Encounters Per Minute:")
    for label, count in two_mice_encounters_per_minute.items():
        print(f'{label}: {count}')

    print("\nOne Mouse Encounters Per Minute:")
    for label, count in one_mouse_encounters_per_minute.items():
        print(f'{label}: {count}')

    total_one_mouse_encounters = total_encounters - total_encounters2
    print(f"\nTotal One Mouse Encounters: {total_one_mouse_encounters}")

