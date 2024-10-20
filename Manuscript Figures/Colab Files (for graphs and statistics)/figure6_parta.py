# -*- coding: utf-8 -*-
"""Figure6_PartA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ep_0szj4tD-ko4vYl2__iKNJGe28VV6t
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch #Line2D, which created legend in colored graphs, does not have a hatch option
from google.colab import drive
from google.colab import files
import os
drive.mount('/content/drive/')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {
    'Hand': [120, 66, 75, 148],
    'TOI': [129, 101, 115, 140]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame for seaborn
df_melted = df.melt(var_name='Method', value_name='Value')

# Set the colors for the bars
palette = {'Hand': 'antiquewhite', 'TOI': 'slategray'}

# Set the figure size
plt.figure(figsize=(10, 6))
plt.rcParams['pdf.fonttype'] = 42 # to work with in illustrator
plt.rcParams['ps.fonttype'] = 42 # to work with in illustrator

# Create the bar plot with error bars
ax = sns.barplot(x='Method', y='Value', data=df_melted, palette=palette, ci='sd', capsize=0.2)

# Overlay the individual data points
sns.stripplot(x='Method', y='Value', data=df_melted, jitter=True, color='black', size=8, alpha=0.6, ax=ax)

# Add a horizontal line for the x-axis
plt.axhline(0, color='black', linewidth=1)

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Customize the plot
plt.title('')
plt.xlabel('')
plt.ylabel('Value')
plt.ylim(0, 200)

# Save file as a PDF to Downloads folder on Mac
file_path = "/content/TOIGroundTruth.pdf"
plt.savefig(file_path)
files.download(file_path)

# Show the plot
plt.show()

# Sample data
data = {
    'Hand': [14, 10, 7, 5],
    'TOI': [11, 6, 4, 5]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame for seaborn
df_melted = df.melt(var_name='Method', value_name='Value')

# Set the colors for the bars
palette = {'Hand': 'antiquewhite', 'TOI': 'slategray'}

# Set the figure size
plt.figure(figsize=(10, 6))
plt.rcParams['pdf.fonttype'] = 42 # to work with in illustrator
plt.rcParams['ps.fonttype'] = 42 # to work with in illustrator

# Create the bar plot with error bars
ax = sns.barplot(x='Method', y='Value', data=df_melted, palette=palette, ci='sd', capsize=0.2)

# Overlay the individual data points
sns.stripplot(x='Method', y='Value', data=df_melted, jitter=True, color='black', size=8, alpha=0.6, ax=ax)

# Add a horizontal line for the x-axis
plt.axhline(0, color='black', linewidth=1)

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Customize the plot
plt.title('')
plt.xlabel('')
plt.ylabel('Value')
plt.ylim(0, 15)

file_path = "/content/TOITwoMiceGroundTruth.pdf"
plt.savefig(file_path)
files.download(file_path)


# Show the plot
plt.show()