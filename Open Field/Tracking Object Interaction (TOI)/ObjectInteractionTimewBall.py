# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:11:33 2024

@author: markd
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load in files generated from createOrderedCenters.py
order_dir = r'.\exp_OrderedCenters'
mouse_and_ball_files = [f for f in os.listdir(order_dir) if f.endswith('.csv')]

compare_shift = 5
min_threshold = 20  # Minimum threshold for significant movement
max_threshold = 40  # Maximum threshold for significant movement

for mouse_and_ball_file in mouse_and_ball_files:
    ordered_csv = os.path.join(order_dir, mouse_and_ball_file)
    df_m = pd.read_csv(ordered_csv)
    
    # Calculating change in distance for ball (using distance formula)
    df_m['db'] = np.sqrt((df_m['bx'].shift(-compare_shift) - df_m['bx'])**2 + 
                          (df_m['by'].shift(-compare_shift) - df_m['by'])**2)
    df_m['db'].fillna(0, inplace=True)
    
    # Calculate difference in distance changes to detect significant movements
    df_m['db_diff'] = abs(df_m['db'] - df_m['db'].shift(1))
    df_m['db_diff'].fillna(0, inplace=True)
    
    # Detect "spikes" in movement
    spikes = df_m['db_diff'] >= min_threshold
    
    # Use the maximum value within each spike to calculate the distance
    significant_movements = []
    spike_start = None
    
    for idx in range(len(spikes)):
        if spikes.iloc[idx] and spike_start is None:
            spike_start = idx
        elif not spikes.iloc[idx] and spike_start is not None:
            spike_end = idx
            max_movement = df_m['db_diff'].iloc[spike_start:spike_end].max()
            if min_threshold <= max_movement <= max_threshold:
                significant_movements.append(max_movement)
            spike_start = None
    
    # Handle the case where a spike continues till the end of the series
    if spike_start is not None:
        max_movement = df_m['db_diff'].iloc[spike_start:].max()
        if min_threshold <= max_movement <= max_threshold:
            significant_movements.append(max_movement)
    
    total_distance = sum(significant_movements) / 1000  # convert pixels to meters (assuming 1000 pixels = 1 meter)
    
    print(f'{mouse_and_ball_file}: Total significant ball distance traveled = {total_distance:.3f} meters')
    
    # Visualize the ball distance changes and threshold range
    plt.figure()
    plt.plot(df_m['db_diff'], label='Ball Distance Changes')
    plt.axhline(y=min_threshold, color='r', linestyle='--', label='Min Threshold')
    plt.axhline(y=max_threshold, color='b', linestyle='--', label='Max Threshold')
    plt.title(f'Ball Distance Changes for {mouse_and_ball_file}')
    plt.xlabel('Frame')
    plt.ylabel('Distance Change')
    plt.legend()
    plt.show()




