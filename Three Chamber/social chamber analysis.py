import h5py
import numpy as np
import os 
#dir = file path
file = (r"/Users/megan_nelson/Downloads/labels.v002.000_Test_C348NSaline-10202023114914.analysis.h5")
#file = os.path.join(path,'C128_L-03162022103232.avi.predictions.000_C128_L-03162022103232.analysis.h5')

#file = name of particular h5 file


with h5py.File(file, "r") as f:
    occupancy_matrix = f['track_occupancy'][:]
    tracks_matrix = f['tracks'][:]
    track_names = f['track_names'][:]

print(occupancy_matrix.shape)
print(tracks_matrix.shape)
#%% nodes = skeleton

filename = file
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
#%%
frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

#%%
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
#%%
locations = fill_missing(locations)
#%%
SNOUT_INDEX = 0
Rear_INDEX = 1
Lear_INDEX = 2
TAILBASE_INDEX = 3
Tailmid_INDEX = 4
Tailtip_INDEX = 5
CENTROID_INDEX = 6

snout_loc = locations[:, SNOUT_INDEX, :, :]
centroid_loc = locations[:, CENTROID_INDEX, :, :]
tailbase_loc = locations[:, TAILBASE_INDEX, :, :]
Rear_loc = locations[:, Rear_INDEX, :, :]
Lear_loc = locations[:, Lear_INDEX, :, :]
Tailmid_loc = locations[:, Tailmid_INDEX, :, :]
Tailtip_loc = locations[:, Tailtip_INDEX, :, :]
#%%
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
#%%
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]
#%%
plt.figure()


plt.plot(centroid_loc[:,0,0], 'k',label='mouse-0') 
#plt.plot(centroid_loc[:,0,1], 'g',label='mouse-1')
#uncomment above to track multiple mice

plt.plot(-1*centroid_loc[:,1,0], 'k') #To check a small time window of 30fps and 5min ex: [0:9000,1,0]
#plt.plot(-1*centroid_loc[:,1,1], 'g')

plt.legend(loc="center right")
plt.title('Centroid locations')
#%% tracks for entire video

plt.figure(figsize=(7,7))
plt.plot(centroid_loc[:,0,0],centroid_loc[:,1,0], 'k',label='mouse-0')

plt.legend()

plt.xlim(100,1200)
#plt.xticks([])

plt.ylim(0,1100)
#plt.yticks([])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Centroid tracks')
plt.gca().set_aspect('equal', adjustable='box')
#%%
import pandas as pd
#%% # tracks out of position? maybe under 200? set all values with x or y coordinates below 400 to nan
allchambers1 = pd.DataFrame({'centroid_x': centroid_loc[:, 0,0], 'centroid_y': centroid_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
allchambers2=allchambers1[(allchambers1['centroid_x'] >= 160) & (allchambers1['centroid_x'] <= 1100)] #define location of box
allchambers=allchambers2[(allchambers2['centroid_y'] >= 160) & (allchambers2['centroid_y'] <= 800)]

plt.scatter(allchambers['centroid_x'],allchambers['centroid_y'], )
plt.gca().set_aspect('equal', adjustable='box')

#%% tracks for first x frames

#for i in range(1): #if multiple mice change 1 to # of mice
#    plt.plot(tracks_matrix[i,0,3,0:1000],tracks_matrix[i,1,3,0:1000]) #3 is a node

plt.title('Centroid tracks')
#%% tracks for first x frames

#for i in range(1): #if multiple mice change 1 to # of mice
#    plt.plot(tracks_matrix[i,0,3,0:1000],tracks_matrix[i,1,3,0:1000]) #3 is a node
    
plt.xlim(100,1100)
#plt.xticks([])

plt.ylim(0,800)
   
plt.gca().set_aspect('equal', adjustable='box')

plt.title('Centroid tracks')

#plt.text(0.5, 1.0, 'Body Center locations')

#%%node tracks and velocity
#%% # Determine time spent and distance traveled 

import pandas as pd
from scipy.spatial import distance
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm

#all tracks
allchambers1 = pd.DataFrame({'centroid_x': centroid_loc[:, 0,0], 'centroid_y': centroid_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.
allchambers2=allchambers1[(allchambers1['centroid_x'] >= 160) & (allchambers1['centroid_x'] <= 1100)] #define location of box
allchambers=allchambers2[(allchambers2['centroid_y'] >= 160) & (allchambers2['centroid_y'] <= 800)]

#total distance traveled
test=allchambers.to_numpy() #change back to array for distance calculation

totaldist = 0 #creates dataframe
for i in range(0, len(test)-1):
    totaldist += distance.euclidean(test[i], test[i+1])*(1/20)*(1/100) #calculates based on 1cm = 10 pixels to meters

#total time in chamber
totaltime = allchambers.shape[0]*(1/60) #calculates time based on 30 frames per second

#chamber 1 tracks
chamb1=allchambers[(allchambers['centroid_x'] >= 160) & (allchambers['centroid_x'] <= 420)] #define location of box


#chamber 1 distance traveled
chamb1t=chamb1.to_numpy()

chamb1dist = 0
for i in range(0, len(chamb1t)-1):
    chamb1dist += distance.euclidean(chamb1t[i], chamb1t[i+1])*(1/20)*(1/100)

#time in chamber 1
chamb1time = chamb1.shape[0]*(1/60)

#chamber 2 tracks
chamb2=allchambers[(allchambers['centroid_x'] >= 500) & (allchambers['centroid_x'] <= 760)] #define location of box

#chamber 2 distance traveled
chamb2t=chamb2.to_numpy()

chamb2dist = 0
for i in range(0, len(chamb2t)-1):
    chamb2dist += distance.euclidean(chamb2t[i], chamb2t[i+1])*(1/20)*(1/100)

#time in chamber 2
chamb2time = chamb2.shape[0]*(1/60)

#chamber 3 tracks
chamb3=allchambers[(allchambers['centroid_x'] >= 860) & (allchambers['centroid_x'] <= 1100)] #define location of box

#chamber 3 distance traveled
chamb3t=chamb3.to_numpy()

chamb3dist = 0
for i in range(0, len(chamb3t)-1):
    chamb3dist += distance.euclidean(chamb3t[i], chamb3t[i+1])*(1/20)*(1/100)

#time in chamber 3
chamb3time = chamb3.shape[0]*(1/60)

#distance travelled crossing into chamber 1
chamb1cross=allchambers[(allchambers['centroid_x'] >= 420) & (allchambers['centroid_x'] <= 500)] #define crossing region
plt.scatter(chamb1cross['centroid_x'],chamb1cross['centroid_y'])

chamb1crosst=chamb1cross.to_numpy()

chamb1crossdist = 0
for i in range(0, len(chamb1crosst)-1):
    chamb1crossdist += distance.euclidean(chamb1crosst[i], chamb1crosst[i+1])*(1/20)*(1/100)

#number of crossings into chamber 1
ch1cross=allchambers[(allchambers['centroid_x'] >= 420) & (allchambers['centroid_x'] <= 500)] #define crossing line
plt.scatter(ch1cross['centroid_x'],ch1cross['centroid_y'])
ch1crossnum = ch1cross.shape[0]

#distance travelled crossing into chamber 3
chamb3cross=allchambers[(allchambers['centroid_x'] >= 760) & (allchambers['centroid_x'] <= 860)] #define crossing region
plt.scatter(chamb3cross['centroid_x'],chamb3cross['centroid_y'])
chamb3crosst=chamb3cross.to_numpy()

chamb3crossdist = 0
for i in range(0, len(chamb3crosst)-1):
    chamb3crossdist += distance.euclidean(chamb3crosst[i], chamb3crosst[i+1])*(1/20)*(1/100)

#number ofcrossings into chamber 3
ch3cross=allchambers[(allchambers['centroid_x'] >= 760) & (allchambers['centroid_x'] <= 860)] #define location of box
plt.scatter(ch3cross['centroid_x'],ch3cross['centroid_y'])
ch3crossnum = ch3cross.shape[0]

plt.savefig(file + "crossings.jpg")

print('Total distance traveled (m):', totaldist)
print('Total time (s):', totaltime)
print('')
print('Chamber 1 distance traveled (m):', chamb1dist)
print('Time spent in chamber 1 (s):', chamb1time) #seconds in defined component of the box
print('')
print('Chamber 2 distance traveled (m):', chamb2dist)
print('Time spent in chamber 2 (s):', chamb2time) #seconds in defined component of the box
print('')
print('Chamber 3 distance traveled (m):', chamb3dist)
print('Time spent in chamber 3 (s):', chamb3time) #seconds in defined component of the box
print('')
print('Number of crossings into chamber 1:', ch1crossnum) #seconds in defined component of the box
print('')
print('Number of crossings into chamber 3:', ch3crossnum) #seconds in defined component of the box
print('')
#%%
#@title
#%%Generate heatmap with scatter

#make heatmaps
x = allchambers['centroid_x']
y = allchambers['centroid_y']

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent
fig, axs = plt.subplots(1, 2)

sigmas = [0, 32]
for ax, s in zip(axs.flatten(), sigmas):
    if s == 0:
        ax.plot(x, y, 'k.', markersize=5)
        ax.set_title("Scatter plot")
        ax.set_xlim(0,1200)
        ax.set_ylim(0,1100)
    else:
        img, extent = myplot(x, y, s)
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        ax.set_xlim(0,1200)
        ax.set_ylim(0,1100)
        ax.set_title("Smoothing with  $sigma$ = %d" % s)
        plt.savefig(file + "centroidscatterandheatmap.jpg")
plt.show()

#%%
#%%Generate line plot in different colors per chamber

#make figure for total chamber tracks
plt.plot(allchambers['centroid_x'],allchambers['centroid_y'],'grey')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Centroid tracks')

#make figure for chamber 1 tracks
plt.plot(chamb1['centroid_x'],chamb1['centroid_y'],'r')
plt.gca().set_aspect('equal', adjustable='box')
#plt.title('Chamber 1 Centroid tracks')

#make figure for chamber 2 tracks
plt.plot(chamb2['centroid_x'],chamb2['centroid_y'],'b')
plt.gca().set_aspect('equal', adjustable='box')
#plt.title('Chamber 2 Centroid tracks')

#make figure for chamber 3 tracks
plt.plot(chamb3['centroid_x'],chamb3['centroid_y'],'g')
plt.gca().set_aspect('equal', adjustable='box')
#plt.title('Chamber 3 Centroid tracks')

plt.savefig(file + "centroidtracks.jpg")
#%%
#%% # Determine time spent and distance traveled around cups

import pandas as pd
from scipy.spatial import distance
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm

#all tracks
allchambers = pd.DataFrame({'snout_x': snout_loc[:, 0,0], 'snout_y': snout_loc[:, 1,0]}) #convert location tracks to dataframe. Centroid chosen here.

#total distance traveled
test=allchambers.to_numpy() #change back to array for distance calculation

totaldist = 0 #creates dataframe
for i in range(0, len(test)-1):
    totaldist += distance.euclidean(test[i], test[i+1])*(1/20)*(1/100) #calculates based on 1cm = 10 pixels to meters

#total time in chamber
totaltime = allchambers.shape[0]*(1/60) #calculates time based on 30 frames per second

#chamber 1 tracks
cup1x=allchambers[allchambers['snout_x'] <= 450] #define location of box
cup1=cup1x[(allchambers['snout_y'] >=400) & (allchambers['snout_y'] <= 600)]

#chamber 1 distance traveled
cup1t=cup1.to_numpy()

cup1dist = 0
for i in range(0, len(cup1t)-1):
    cup1dist += distance.euclidean(cup1t[i], cup1t[i+1])*(1/20)*(1/100)

#time in chamber 1
cup1time = cup1.shape[0]*(1/60)

#chamber 3 tracks
cup3x=allchambers[(allchambers['snout_x'] >= 850) & (allchambers['snout_x'] <= 120000)] #define location of box
cup3=cup3x[(allchambers['snout_y'] >=400) & (allchambers['snout_y'] <= 600)]

#chamber 3 distance traveled
cup3t=cup3.to_numpy()

cup3dist = 0
for i in range(0, len(cup3t)-1):
    cup3dist += distance.euclidean(cup3t[i], cup3t[i+1])*(1/20)*(1/100)

#time in chamber 3
cup3time = cup3.shape[0]*(1/60)

print('Cup 1 distance traveled (m):', cup1dist)
print('Time spent around cup 1 (s):', cup1time) #seconds in defined component of the box
print('')
print('Cup 3 distance traveled (m):', cup3dist)
print('Time spent around cup 3 (s):', cup3time) #seconds in defined component of the box
print('')
#%%
#plot tracks around the cups
plt.xlim(100,1200)
plt.ylim(0,1100)
plt.plot(cup1['snout_x'],cup1['snout_y'],'r')
plt.plot(cup3['snout_x'],cup3['snout_y'],'g')
#%%
import numpy as np
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

Video_FILE = (r"/Users/megan_nelson/Downloads/Test_C348NSaline-10202023114914.avi")

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


frame = get_frame(Video_FILE,26760)
print('shape is', frame.shape)
print('pixel at (60,21)',frame[60,21,:])
print('pixel at (120,10)',frame[120,10,:])

plt.figure(figsize=(15,15)) #starting here you need trackanalysis_social.py code
plt.plot(centroid_loc[:,0,0],centroid_loc[:,1,0], 'k')

plt.legend()

plt.xticks(np.arange(0, len(centroid_loc[:,0,0])+1, 20))
plt.xlim(100,1200)
plt.xticks(fontsize=10)
plt.xticks(rotation=45)

plt.yticks(np.arange(0, len(centroid_loc[:,0,0])+1, 20))
plt.ylim(0,1100)
#plt.yticks([])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Centroid Center tracks')
plt.imshow(frame)
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig(file + "movieandtracks.jpg")
#%%