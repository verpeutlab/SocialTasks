# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:59:46 2021

@author: jverpeut
"""

import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm

#%%
filename = (r'/Users/megan_nelson/Desktop/CURRENT_DESKTOP/Verpeut_Lab/Honors_Thesis/Experimental_Open_Field/OF_C346N-10062023120112.avi_1_22.avi.predictions.000_OF_C346N-10062023120112.analysis.h5')

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

nose_loc = locations[:, NOSE_INDEX, :, :]
EarR_loc = locations[:, EarR_INDEX, :, :]
TailBase_loc = locations[:, TailBase_INDEX, :, :]

#%% Plot X.Y coordinates of a node per mouse (mouse 1 = 0, mouse 2 = 1) 
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]

plt.figure()

#[frames,xandy,mouse]
plt.plot(TailBase_loc[:,0,0], 'k',label='mouse-0') #To check a small time window of 30fps and 5min ex: [0:9000,0,0]
plt.plot(TailBase_loc[:,0,1], 'g',label='mouse-1') #k is black and g is green: change as you wish

#[frames,yandy,mouse]
plt.plot(-1*TailBase_loc[:,1,0], 'k') #To check a small time window of 30fps and 5min ex: [0:9000,1,0]
plt.plot(-1*TailBase_loc[:,1,1], 'g')

plt.legend(loc="center right")
plt.title('Tail Base locations in X-Y coordinate space')
plt.xlabel('Frames')
plt.ylabel('Distance (pixels)')

#%% tracks for entire video of mouse-0

plt.figure(figsize=(7,7))
plt.plot(TailBase_loc[:,0,0],TailBase_loc[:,1,0], 'k')

plt.legend()

plt.xlim(250,1000)
plt.ylim(90,850)

plt.gca().set_aspect('equal', adjustable='box')
plt.title('Tail Base tracks')
plt.gca().set_aspect('equal', adjustable='box')

plt.ylabel('Distance (pixels)')
plt.xlabel('Distance (pixels)')

#%% tracks for entire video of mouse-1

plt.figure(figsize=(7,7))
plt.plot(TailBase_loc[:,0,1],TailBase_loc[:,1,1], 'g')

plt.legend()

plt.xlim(250,1000)
plt.ylim(90,850)

plt.gca().set_aspect('equal', adjustable='box')
plt.title('Tail Base tracks')
plt.gca().set_aspect('equal', adjustable='box')

plt.ylabel('Distance (pixels)')
plt.xlabel('Distance (pixels)')


#%% # tracks out of position? maybe under 400? set all values with x or y coordinates below 400 to nan
# tracks_matrix[tracks_matrix < 400] = np.nan

# plt.figure()
# plt.plot(tracks_matrix[0,0,3,:],tracks_matrix[0,1,3,:], 'k') #3 is a node
# plt.gca().set_aspect('equal', adjustable='box')

#plt.text(0.5, 1.0, 'Body Center locations')
#%% tracks for first x frames

#for i in range(1): #if multiple mice change 1 to # of mice
 #   plt.plot(tracks_matrix[i,0,3,0:1000],tracks_matrix[i,1,3,0:1000]) #3 is a node
    
#plt.xlim(100,1200)
#plt.xticks([])

#plt.ylim(0,1100)
   
#plt.gca().set_aspect('equal', adjustable='box')

#plt.title('Nose Center tracks')

#plt.text(0.5, 1.0, 'Body Center locations')

#%%velocity calculation per node

from scipy.signal import savgol_filter

def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    node_loc_vel = np.zeros_like(node_loc)
    
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel

#Calculate velocity for tail base node
TB_vel_mouse0 = smooth_diff(TailBase_loc[:, :, 0])
TB_vel_mouse1 = smooth_diff(TailBase_loc[:, :, 1])

#%%visualization of velocity mouse-0

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(211)
ax1.plot(TailBase_loc[:, 0, 0], 'k', label='x')
ax1.plot(-1*TailBase_loc[:, 1, 0], 'k', label='y')
ax1.legend()
ax1.set_xticks([])
ax1.set_title('Tail Base velocity in X-Y coordinate space')
ax1.set_xlabel('Frames')
ax1.set_ylabel('Distance (pixels)')

ax2 = fig.add_subplot(212, sharex=ax1)
im1=ax2.imshow(TB_vel_mouse0[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10)
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right',size='1%',pad=0.05)
fig.colorbar(im1, cax,orientation='vertical')
ax2.set_yticks([])
ax2.set_title('Tail Base Velocity')

#%%visualization of velocity mouse-1
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(211)
ax1.plot(TailBase_loc[:, 0, 1], 'g', label='x')
ax1.plot(-1*TailBase_loc[:, 1, 1], 'g', label='y')
ax1.legend()
ax1.set_xticks([])
ax1.set_title('Tail Base velocity in X-Y coordinate space')
ax1.set_xlabel('Frames')
ax1.set_ylabel('Distance (pixels)')

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(TB_vel_mouse1[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10)
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right',size='1%',pad=0.05)
fig.colorbar(im1, cax,orientation='vertical')
ax2.set_yticks([])
ax2.set_title('Tail Base Velocity')

#%% tracks with velocity on top for entire video for mouse-0

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax1.plot(TailBase_loc[:, 0, 0], TailBase_loc[:, 1, 0], 'k')
ax1.set_xlim(250,1000)
ax1.set_xticks([])
ax1.set_ylim(90,850)
ax1.set_yticks([])
ax1.set_title('Tail Base tracks')

kp = TB_vel_mouse0 #all track
vmin = 0
vmax = 10

ax2 = fig.add_subplot(122)
ax2.scatter(TailBase_loc[:,0,0], TailBase_loc[:,1,0], c=kp, s=4, vmin=vmin, vmax=vmax)
ax2.set_xlim(250,1000)
ax2.set_xticks([])
ax2.set_ylim(90,850)
ax2.set_yticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right',size='2%',pad=0.05)
fig.colorbar(im1, cax,orientation='vertical')
ax2.set_title('Nose tracks colored by magnitude of fly speed')

#%% tracks with velocity on top for entire video for mouse-1

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax1.plot(TailBase_loc[:, 0, 1], TailBase_loc[:, 1, 1], 'g')
ax1.set_xlim(250,1000)
ax1.set_xticks([])
ax1.set_ylim(90,850)
ax1.set_yticks([])
ax1.set_title('Tail Base tracks')

kp = TB_vel_mouse0 #all track
vmin = 0
vmax = 10

ax2 = fig.add_subplot(122)
ax2.scatter(TailBase_loc[:,0,1], TailBase_loc[:,1,1], c=kp, s=4, vmin=vmin, vmax=vmax)
ax2.set_xlim(250,1000)
ax2.set_xticks([])
ax2.set_ylim(90,850)
ax2.set_yticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right',size='2%',pad=0.05)
fig.colorbar(im1, cax,orientation='vertical')
ax2.set_title('Nose tracks colored by magnitude of fly speed')

#%%covariance in velocities between both mice

def corr_roll(datax, datay, win):
    """
    datax, datay are the two timeseries to find correlations between
    
    win sets the number of frames over which the covariance is computed
    
    """
    
    s1 = pd.Series(datax)
    s2 = pd.Series(datay)
    
    return np.array(s2.rolling(win).corr(s1))

win = 1000

cov_vel = corr_roll(TB_vel_mouse0, TB_vel_mouse1,win)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15,6))
ax[0].plot(TB_vel_mouse0, 'k', label='mouse-0')
ax[0].plot(TB_vel_mouse1, 'g', label='mouse-1')
ax[0].legend()
ax[0].set_title('Forward Velocity')

ax[1].plot(cov_vel, 'c', markersize=1)
ax[1].set_ylim(-1.2, 1.2)
ax[1].set_title('Covariance')

fig.tight_layout()

#%% calculate each node velocity per frame

def instance_node_velocities(instance_idx):
    mouse_node_locations = locations[:, :, :, instance_idx]
    mouse_node_velocities = np.zeros((frame_count, node_count))

    for n in range(0, node_count):
        mouse_node_velocities[:, n] = smooth_diff(mouse_node_locations[:, n, :])
    
    return mouse_node_velocities

def plot_instance_node_velocities(instance_idx, node_velocities):
    plt.figure(figsize=(20,8))
    plt.imshow(node_velocities.T, aspect='auto', vmin=0, vmax=10, interpolation="nearest")
    plt.xlabel('frames')
    plt.ylabel('nodes')
    plt.yticks(np.arange(node_count), node_names, rotation=20);
    plt.title(f'Mouse {instance_idx} node velocities')
    
#%%node velocity mouse-0
    
mouse_ID = 0
mouse_node_velocities = instance_node_velocities(mouse_ID)
plot_instance_node_velocities(mouse_ID, mouse_node_velocities)
cax=plt.axes([0.905,0.13,0.015,0.75])
plt.colorbar(cax=cax)

#%%s behavior clusters for mouse-0

#RUN ONLY AFTER LINES 354-358
nstates = 5
km = KMeans(n_clusters=nstates)

labels = km.fit_predict(mouse_node_velocities)


fig = plt.figure(figsize=(20, 12))

ax1 = fig.add_subplot(211)
ax1.imshow(mouse_node_velocities.T, aspect="auto", vmin=0, vmax=10, interpolation="nearest")
ax1.set_xlabel("Frames")
ax1.set_ylabel("Nodes")
ax1.set_yticks(np.arange(node_count))
ax1.set_yticklabels(node_names);
ax1.set_title(f"Mouse {mouse_ID} node velocities")
ax1.set_xlim(0,frame_count)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right',size='2%',pad=0.05)
fig.colorbar(im1, cax,orientation='vertical')

ax2 = fig.add_subplot(212,sharex=ax1)
im2=ax2.imshow(labels[None, :], aspect="auto", cmap="tab10", interpolation="nearest")
ax2.set_xlabel("Frames")
ax2.set_yticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right',size='2%',pad=0.05)
fig.colorbar(im2, cax,orientation='vertical')
ax2.set_title("Ethogram (colors = clusters)");

unique, counts = np.unique(labels, return_counts=True)
counts_sorted = np.sort(counts)
#print("counts_sorted", counts_sorted)

behaviors = labels.tolist() #time in a behavior in seconds
print ("Count for 0", behaviors.count(0)/30)
print ("Count for 1", behaviors.count(1)/30)
print ("Count for 2", behaviors.count(2)/30)
print ("Count for 3", behaviors.count(3)/30)
print ("Count for 4", behaviors.count(4)/30)


from sklearn.decomposition import PCA

# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(mouse_node_velocities)

# Transfor the scaled data to the new PCA space
X_reduced = pca.transform(mouse_node_velocities)

# Convert to a data frame
X_reduceddf = pd.DataFrame(X_reduced, columns=['PC1','PC2'])
X_reduceddf['cluster'] = labels
X_reduceddf.head()

#%%node velocity mouse-1
mouse_ID = 1
mouse_node_velocities = instance_node_velocities(mouse_ID)
plot_instance_node_velocities(mouse_ID, mouse_node_velocities)
cax=plt.axes([0.905,0.13,0.015,0.75])
plt.colorbar(cax=cax)

#%%s behavior clusters for mouse-0
#RUN ONLY AFTER LINES 389-394
nstates = 5
km = KMeans(n_clusters=nstates)

labels = km.fit_predict(mouse_node_velocities)

fig = plt.figure(figsize=(20, 12))

ax1 = fig.add_subplot(211)
ax1.imshow(mouse_node_velocities.T, aspect="auto", vmin=0, vmax=10, interpolation="nearest")
ax1.set_xlabel("Frames")
ax1.set_ylabel("Nodes")
ax1.set_yticks(np.arange(node_count))
ax1.set_yticklabels(node_names);
ax1.set_title(f"Mouse {mouse_ID} node velocities")
ax1.set_xlim(0,frame_count)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right',size='2%',pad=0.05)
fig.colorbar(im1, cax,orientation='vertical')

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.imshow(labels[None, :], aspect="auto", cmap="tab10", interpolation="nearest")
ax2.set_xlabel("Frames")
ax2.set_yticks([])
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right',size='2%',pad=0.05)
fig.colorbar(im2, cax,orientation='vertical')
ax2.set_title("Ethogram (colors = clusters)");

behaviors = labels.tolist() #time in a behavior in seconds
print ("Count for 0", behaviors.count(0)/30)
print ("Count for 1", behaviors.count(1)/30)
print ("Count for 2", behaviors.count(2)/30)
print ("Count for 3", behaviors.count(3)/30)
print ("Count for 4", behaviors.count(4)/30)

# Create a PCA model to reduce our data to 2 dimensions for visualisation
pca = PCA(n_components=2)
pca.fit(mouse_node_velocities)

# Transfor the scaled data to the new PCA space
Y_reduced = pca.transform(mouse_node_velocities)

# Convert to a data frame
Y_reduceddf = pd.DataFrame(X_reduced, columns=['PC1','PC2'])
Y_reduceddf['cluster'] = labels
Y_reduceddf.head()

#%% Additional Analysis: Traveled distance in entire chamber and inner chamber. Time spent in inner chamber

#Mouse-0
#all tracks TOTAL
openField_mouse0 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,0], 'centroid_y': TailBase_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse0[(openField_mouse0['centroid_x'] >= 280) & (openField_mouse0['centroid_x'] <= 980)] #define location of box
b=a[(a['centroid_y'] >= 100) & (a['centroid_y'] <= 800)]

#total distance traveled
totaldisttravel_mouse0=b.to_numpy() #change back to array for distance calculation

disttravel_mouse0 = 0 #creates dataframe
for i in range(0, len(totaldisttravel_mouse0)-1):
    disttravel_mouse0 += distance.euclidean(totaldisttravel_mouse0[i], totaldisttravel_mouse0[i+1])*(1/20)*(1/100) #calculates based on 1cm = 10 pixels to meters- NEED TO FIX
# MRN Edited line above to change 1/20 to 1/10
#total time in chamber
totaltime_mouse0 = b.shape[0]*(1/60) #calculates time based on 60 frames per second

#all tracks INNER
INNER_mouse0 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,0], 'centroid_y': TailBase_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
a=INNER_mouse0[(INNER_mouse0['centroid_x'] >= 455) & (INNER_mouse0['centroid_x'] <= 630)] #define location of box
b=a[(a['centroid_y'] >= 275) & (a['centroid_y'] <= 450)]

#total distance traveled in INNER
INNERdisttravel_mouse0=b.to_numpy() #change back to array for distance calculation

INNERtravel_mouse0 = 0 #creates dataframe
for i in range(0, len(INNERdisttravel_mouse0)-1):
    INNERtravel_mouse0 += distance.euclidean(INNERdisttravel_mouse0[i], INNERdisttravel_mouse0[i+1])*(1/20)*(1/100) #calculates based on 1cm = 10 pixels to meters- NEED TO FIX
    
#total time in INNER
INNERtime_M0 = b.shape[0]*(1/60) #calculates time based on 60 frames per second

#Mouse-1
#all tracks
openField_mouse1 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,1], 'centroid_y': TailBase_loc[:, 1,1]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse1[(openField_mouse1['centroid_x'] >= 280) & (openField_mouse1['centroid_x'] <= 980)] #define location of box
b=a[(a['centroid_y'] >= 100) & (a['centroid_y'] <= 800)]

#total distance traveled
totaldisttravel_mouse1=b.to_numpy() #change back to array for distance calculation

#total time in chamber
totaltime_mouse1 = b.shape[0]*(1/60) #calculates time based on 60 frames per second

disttravel_mouse1 = 0 #creates dataframe
for i in range(0, len(totaldisttravel_mouse1)-1):
    disttravel_mouse1 += distance.euclidean(totaldisttravel_mouse1[i], totaldisttravel_mouse1[i+1])*(1/20)*(1/100) #calculates based on 1cm = 10 pixels to meters- NEED TO FIX
# MRN changed 1/20 in previous line to 1/8.5
#all tracks INNER
INNER_mouse1 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,1], 'centroid_y': TailBase_loc[:, 1,1]}) #convert location tracks to dataframe. Centroid chosen here.
a=INNER_mouse1[(INNER_mouse1['centroid_x'] >= 455) & (INNER_mouse1['centroid_x'] <= 630)]
b1=a[(a['centroid_y'] >= 275) & (a['centroid_y'] <= 450)]

#total distance traveled in INNER
INNERdisttravel_mouse1=b1.to_numpy() #change back to array for distance calculation

INNERtravel_mouse1 = 0 #creates dataframe
for i in range(0, len(INNERdisttravel_mouse1)-1):
    INNERtravel_mouse1 += distance.euclidean(INNERdisttravel_mouse1[i], INNERdisttravel_mouse1[i+1])*(1/20)*(1/100) #calculates based on 1cm = 10 pixels to meters- NEED TO FIX

#total time in INNER
INNERtime_M1 = b1.shape[0]*(1/60) #calculates time based on 60 frames per second

print('M0: Total distance traveled (m):', disttravel_mouse0)
print('M0: Total time spent in chamber (s)', totaltime_mouse0)
print('M0: INNER distance traveled (m):', INNERtravel_mouse0)
print('M0: INNER time spent (s):', INNERtime_M0)
print('M1: Total distance traveled (m):', disttravel_mouse1)
print('M1: Total time spent in chamber (s)', totaltime_mouse1)
print('M1: INNER distance traveled (m):', INNERtravel_mouse1)
print('M1: INNER time spent (s):', INNERtime_M1)
print('')
#%% Additional Analysis: Time spent in each corner of arena for M0 only

#Mouse-0
#front left corner
openField_mouse0 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,0], 'centroid_y': TailBase_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse0[(openField_mouse0['centroid_x'] >= 280) & (openField_mouse0['centroid_x'] <= 455)] #define location of box
b=a[(a['centroid_y'] >= 100) & (a['centroid_y'] <= 175)]

#total distance traveled
frontleftcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
frontleft = frontleftcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

#front right corner
openField_mouse0 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,0], 'centroid_y': TailBase_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse0[(openField_mouse0['centroid_x'] >= 630) & (openField_mouse0['centroid_x'] <= 980)] #define location of box
b=a[(a['centroid_y'] >= 100) & (a['centroid_y'] <= 175)]

#total distance traveled
frontrightcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
frontright = frontrightcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

#back right corner
openField_mouse0 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,0], 'centroid_y': TailBase_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse0[(openField_mouse0['centroid_x'] >= 630) & (openField_mouse0['centroid_x'] <= 980)] #define location of box
b=a[(a['centroid_y'] >= 450) & (a['centroid_y'] <= 800)]

#total distance traveled
backrightcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
backright = backrightcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

#back left corner
openField_mouse0 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,0], 'centroid_y': TailBase_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse0[(openField_mouse0['centroid_x'] >= 280) & (openField_mouse0['centroid_x'] <= 455)] #define location of box
b=a[(a['centroid_y'] >= 450) & (a['centroid_y'] <= 800)]

#total distance traveled
backlefttcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
backleft = backleftcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

totaltime = backleft + backright + frontright + frontleft

print('M0: Total time spent in corners (s):', totaltime)
print('')

#%% Additional Analysis: Time spent in each corner of arena for M1 only

#Mouse-0
#front left corner
openField_mouse1 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,1], 'centroid_y': TailBase_loc[:, 1,1]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse1[(openField_mouse1['centroid_x'] >= 280) & (openField_mouse1['centroid_x'] <= 455)] #define location of box
b=a[(a['centroid_y'] >= 100) & (a['centroid_y'] <= 175)]

#total distance traveled
frontleftcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
frontleft = frontleftcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

#front right corner
openField_mouse1 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,1], 'centroid_y': TailBase_loc[:, 1,1]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse1[(openField_mouse1['centroid_x'] >= 630) & (openField_mouse1['centroid_x'] <= 980)] #define location of box
b=a[(a['centroid_y'] >= 100) & (a['centroid_y'] <= 175)]

#total distance traveled
frontrightcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
frontright = frontrightcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

#back right corner
openField_mouse1 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,1], 'centroid_y': TailBase_loc[:, 1,1]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse1[(openField_mouse1['centroid_x'] >= 630) & (openField_mouse1['centroid_x'] <= 980)] #define location of box
b=a[(a['centroid_y'] >= 450) & (a['centroid_y'] <= 800)]

#total distance traveled
backrightcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
backright = backrightcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

#back left corner
openField_mouse1 = pd.DataFrame({'centroid_x': TailBase_loc[:, 0,1], 'centroid_y': TailBase_loc[:, 1,1]}) #convert location tracks to dataframe. Centroid chosen here.
a=openField_mouse1[(openField_mouse1['centroid_x'] >= 280) & (openField_mouse1['centroid_x'] <= 455)] #define location of box
b=a[(a['centroid_y'] >= 450) & (a['centroid_y'] <= 800)]

#total distance traveled
backlefttcorner=b.to_numpy() #change back to array for distance calculation

#total time in chamber
backleft = backleftcorner=b.shape[0]*(1/60) #calculates time based on 60 frames per second

totaltime = backleft + backright + frontright + frontleft

print('M1: Total time spent in corners (s):', totaltime)
print('')

#%%Generate heatmap with scatter for M0

#make heatmaps
x = openField_mouse0['centroid_x']
y = openField_mouse0['centroid_y']

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent
fig, axs = plt.subplots(1, 2)

sigmas = [0, 32]
for ax, s in zip(axs.flatten(), sigmas):
    if s == 0:
        ax.plot(x, y, 'k.', markersize=1)
        ax.set_title("Scatter plot")
        ax.set_xlim(250,1000)
        ax.set_ylim(100,880)
    else:
        img, extent = myplot(x, y, s)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        ax.set_xlim(250,1000)
        ax.set_ylim(100,880)
        ax.set_title("Smoothing with  $sigma$ = %d" % s)
        #plt.savefig(file + "centroidscatterandheatmap.jpg")
plt.show()

#%%Generate heatmap with scatter for M1

#make heatmaps
x = openField_mouse1['centroid_x']
y = openField_mouse1['centroid_y']

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent
fig, axs = plt.subplots(1, 2)

sigmas = [0, 32]
for ax, s in zip(axs.flatten(), sigmas):
    if s == 0:
        ax.plot(x, y, 'k.', markersize=1)
        ax.set_title("Scatter plot")
        ax.set_xlim(250,1000)
        ax.set_ylim(100,880)
    else:
        img, extent = myplot(x, y, s)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        ax.set_xlim(250,1000)
        ax.set_ylim(100,880)
        ax.set_title("Smoothing with  $sigma$ = %d" % s)
        #plt.savefig(file + "centroidscatterandheatmap.jpg")
plt.show()


#%%visualize tracks with video frame

import cv2
import numpy as np
import matplotlib.pyplot as plt

Video_FILE = r"/Users/megan_nelson/Desktop/CURRENT_DESKTOP/Verpeut_Lab/Honors_Thesis/Experimental_Open_Field/OF_C346N-10062023120112.avi"

def get_frames(filename):
    video=cv2.VideoCapture(filename)
    while video.isOpened():
        rete,frame=video.read()
        if rete:
            yield frame
        else:
            break
        video.release()
        yield None

for f in get_frames('Video_FILE'):
    if f is None: break
    cv2.imshow('frame',f)
    if cv2.waitKey(10) == 40: 
        break
cv2.destroyAllWindows()

def get_frame(filename,index):
    counter=0
    video=cv2.VideoCapture(filename)
    while video.isOpened():
        rete,frame=video.read()
        if rete:
            if counter==index:
                return frame
            counter +=1
        else:
            break
    video.release()
    return None


frame = get_frame(Video_FILE,45)
print('shape is', frame.shape)
print('pixel at (60,21)',frame[60,21,:])
print('pixel at (120,10)',frame[120,10,:])

plt.figure(figsize=(15,15)) 
plt.plot(TailBase_loc[:,0,0],TailBase_loc[:,1,0], 'k')

plt.legend()

plt.xticks(np.arange(0, len(TailBase_loc[:,0,0])+1, 20))
plt.xlim(240,1000)
plt.xticks(fontsize=10)
plt.xticks(rotation=45)

plt.ylim(100,850)
#plt.yticks([])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Tail Base tracks of mouse-0')
plt.imshow(frame)
plt.gca().set_aspect('equal', adjustable='box')

#%%
# Create image of all nodes for all mice
# Define the frame number you want to plot (change this to your desired frame)
frame_number = 51095
# Create a list of node names
node_names = [
    "EarL",
    "EArR",
    "Nose",
    "SpineN",
    "SpineF",
    "SpineK",
    "TailBase",
    "TailMid",
    "TailEnd",
    "ShoulderL",
    "ShoulderR",
    "KneeL",
    "KneeR"
]
# Create a color map for differentiating between animals
colors = ['b', 'g']
# Plot all nodes for all three animals on a single plot
plt.figure(figsize=(10, 10))
for node_index, node_name in enumerate(node_names):
    plt.title(f'Node positions at frame {frame_number}')
    plt.xlabel('X-coordinate (pixels)')
    plt.ylabel('Y-coordinate (pixels')
    for i in range(2):  # Loop through each animal - I changed 3 to 2 MN
        x = locations[frame_number, node_index, 0, i]
        y = locations[frame_number, node_index, 1, i]
        plt.scatter(x, y, label=f'{node_name} - Mouse {i + 1}', color=colors[i])
# Add legend
#plt.legend(False) commented this out MN
# Show the plot
plt.show()
