#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:00:05 2023

@author: megan_nelson
"""

import h5py
import numpy as np
#import xarray as xr
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
import csv

#%%
filename = (r'/Users/megan_nelson/Desktop/Three_Chamber_SOLUR_Figures/Open_Field_SOLUR_Figures/Open_Field_Videos/OF_C257R-12062022102746.avi_1_22.avi.predictions.000_OF_C257R-12062022102746.analysis.h5')

with h5py.File(filename, 'r') as f:
    occupancy_matrix = f['track_occupancy'][:]
    tracks_matrix = f['tracks'][:]
    track_names = f['track_names'][:]

print(occupancy_matrix.shape)
print(tracks_matrix.shape)
#%% nodes = skeleton

with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()

#%% file information for frames, node, instance or number of mice
frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

#%% reformat for missing data
from scipy.interpolate import interp1d

def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

locations = fill_missing(locations)
#%% Create dictionary of nodes

#I only add 3 of 12 of your nodes. It will depend on what you want to track.
NOSE_INDEX = 0
EarR_INDEX = 2
TailBase_INDEX = 6
SpineM_INDEX= 4

nose_loc = locations[:, NOSE_INDEX, :, :]
EarR_loc = locations[:, EarR_INDEX, :, :]
TailBase_loc = locations[:, TailBase_INDEX, :, :]
SpineM_loc = locations[:, SpineM_INDEX, :, :]
print(SpineM_loc[1:10])

# np.savetxt('C249_LR_SpineM.csv', SpineM_loc, delimiter = ',') does not work because 3D numpy array
# stacked = pd.Panel(SpineM_loc.swapaxes(1,2)).to_frame().stack().reset_index()
# stacked.to_csv('stacked.csv', index=False) # had to comment these lines because Panel() takes no arguments

#df_3d = SpineM_loc.to_dataframe()
# File Name: OF_C249LR_coordinates

with open('OF_C257R-12062022102746.csv','w') as file:
    for frame in range(0,len(SpineM_loc)):
        temp = '{},{},{},{},{}'.format(frame,SpineM_loc[frame,0,0],SpineM_loc[frame,1,0],SpineM_loc[frame,0,1],SpineM_loc[frame,1,1]) 
        file.write(temp)
        file.write('\n')
