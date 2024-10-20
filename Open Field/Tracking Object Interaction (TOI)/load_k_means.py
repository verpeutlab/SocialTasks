# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:44:06 2023

This script reformats the k-means output from FineTracker.py.
This is the fourth script ran in TOI code.
"""

import os
import csv
import pandas
import matplotlib.pyplot as plt

# images are     1024 x 1280

# 504, 609   :   504, 1024 - 609 = 504, 415   switch  415, 504
# 379, 385   :   379, 1024 - 385 = 379, 639   switch  639, 379
# what if
# 504, 379
# 609, 385


# 616, 374
# 512, 369




kmean_dir  = r'.\new_KMeansOutput'
# Import k-means output files to reformat them. Want to rearrange the columns so that they are in the same format as SLEAP files that will be used in the next script.
kmeans_files = [f for f in os.listdir(kmean_dir) if f.endswith('.csv')]

for kmeans_file in kmeans_files:
    km_csv = os.path.join(kmean_dir, kmeans_file)
    
    df = pandas.read_csv(km_csv, names=['frame','x1','y1','x2','y2','x3','y3'])
    print(df)
    
    ax = df.plot.scatter(x='frame', y='x1', c='Blue')
    df.plot.scatter(x='frame', y='x2', c='Green', ax=ax)
    df.plot.scatter(x='frame', y='x3', c='Red', ax=ax)
    
    break