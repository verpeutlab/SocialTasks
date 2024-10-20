# -*- coding: utf-8 -*-
"""Figure4Code_PartsBCD.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14dQzlC8OYd_lB3Uxx_r86t50bkyOEpR2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from google.colab import drive
from google.colab import files
import os
drive.mount('/content/drive/')
csv = pd.read_csv('/content/drive/My Drive/Female_StackedBar.csv')
print(csv)

# Plot data as stacked bar plot
csv_percent = csv.copy()
# csv_percent.drop(columns=['Side Side'], inplace=True)
count_columns = ['Side Side', 'Paired Exploration', 'Pursuit', 'Nose Nose', 'Anogenital Sniffing']
csv_percent[count_columns] = csv_percent[count_columns].apply(lambda x: (x / x.sum()) * 100, axis=1)
csv_percent['Mouse Name'] = csv_percent['Mouse Name'].replace({
    'C262_L': '1',
    'C262_R': '2',
    'C274_L': '3',
    'C292_L': '4',
    'C296_LLR': '5',
    'C296_RRL': '6',
    'C302_L': '7',
    'C308_L': '8'
})

# Create the plot with increased figure size
fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size (width, height)
csv_percent.plot(x='Mouse Name', kind='bar', stacked=True, color=['black', 'dimgray', 'gray', 'darkgray', 'gainsboro'], ax=ax)

# Customize the plot
ax.legend(loc='upper left', frameon=False)  # Move the legend inside the graph in the top left position
ax.set_xlabel('Mouse')
ax.set_ylabel('Interactive Behaviors (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylim(0, 125)
plt.xticks(rotation=0)

# Save the file as a PDF
file_path = "/content/StackedFemaleNumBehaviors.pdf"
plt.savefig(file_path, bbox_inches='tight')
files.download(file_path)

# Show the plot
plt.show()

csv2 = pd.read_csv('/content/drive/My Drive/Male_StackedBar.csv')
print(csv2)

# plot data as stacked bar plot
csv_percent = csv2.copy()
# csv_percent.drop(columns=['Side Side'], inplace = True)
count_columns = ['Paired Exploration', 'Pursuit', 'Nose Nose', 'Anogenital Sniffing']
count_columns = ['Side Side', 'Paired Exploration', 'Pursuit', 'Nose Nose', 'Anogenital Sniffing']
csv_percent[count_columns] = csv_percent[count_columns].apply(lambda x: (x / x.sum()) * 100, axis=1)

csv_percent['Mouse Name'] = csv_percent['Mouse Name'].replace({
    'C249_LR': '1',
    'C249_RR': '2',
    'C255_L': '3',
    'C257_L': '4',
    'C257_R': '5',
    'C289_L': '6',
    'C303_L': '7',
    'C313_L': '8'
})


# Create the plot with increased figure size
fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size (width, height)
csv_percent.plot(x='Mouse Name', kind='bar', stacked=True, color=['black', 'dimgray', 'gray', 'darkgray', 'gainsboro'], ax=ax)

# Customize the plot
ax.legend(loc='upper left', frameon=False)  # Move the legend inside the graph in the top left position
ax.set_xlabel('Mouse')
ax.set_ylabel('Interactive Behaviors (%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylim(0, 125)
plt.xticks(rotation=0)

# Save the file as a PDF
file_path = "/content/StackedMaleNumBehaviors.pdf"
plt.savefig(file_path, bbox_inches='tight')
files.download(file_path)

# Show the plot
plt.show()

csv_concat = pd.concat([csv, csv2])
# csv_concat.drop(columns=['Side Side'], inplace = True)
print(csv_concat)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
##New bar graphs with male on left and female on right
# Assuming df is already loaded and cleaned

# Melt the DataFrame to long format
df = csv_concat
# Melt the DataFrame to long format
df_melted = pd.melt(df, id_vars=['Mouse Name', 'Sex'], var_name='Behavior', value_name='Count')

# Sort behaviors alphabetically
df_melted['Behavior'] = pd.Categorical(df_melted['Behavior'], categories=sorted(df_melted['Behavior'].unique()), ordered=True)

# Split the data into male and female
df_male = df_melted[df_melted['Sex'] == 'M'].copy()
df_female = df_melted[df_melted['Sex'] == 'F'].copy()

# Invert the counts for males to place them on the left
df_male['Count'] = -df_male['Count']

# Plot using Seaborn
plt.figure(figsize=(10, 6))
plt.rcParams['pdf.fonttype'] = 42 #to work with in illustrator
plt.rcParams['ps.fonttype'] = 42 #to work with in illustrator

# Create horizontal bar plot for males and females
sns.barplot(data=df_male, y='Behavior', x='Count', color='green', ci='sd', capsize=0.1, orient='h', label='Male')
sns.barplot(data=df_female, y='Behavior', x='Count', color='magenta', ci='sd', capsize=0.1, orient='h', label='Female')

# Individual data points
sns.stripplot(data=df_male, y='Behavior', x='Count', color='black', size=5, alpha=0.7, zorder=1, orient='h')
sns.stripplot(data=df_female, y='Behavior', x='Count', color='black', size=5, alpha=0.7, zorder=1, orient='h')

plt.ylabel('')
plt.xlabel('Total Interactive Behaviors')
plt.title('')

# Remove top and right spines so that only x and y axes are showing
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

# Set the x-axis limits to ensure bars are correctly positioned
max_count = max(df_female['Count'].max(), -df_male['Count'].min())
plt.xlim(-max_count, max_count)

# Define custom legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Female', markerfacecolor='magenta', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Male', markerfacecolor='green', markersize=10)
]

# Create legend and place it in the top left corner
plt.legend(handles=legend_elements, loc='upper right', frameon=False)

# Save file as a PDF
file_path = "/content/MalvsFemaleNumBehaviorsHoriz.pdf"
plt.savefig(file_path)
files.download(file_path)

# Show the plot
plt.show()

csv3 = pd.read_csv('/content/drive/My Drive/Male_MedianBehaviorTime.csv')
csv4 = pd.read_csv('/content/drive/My Drive/Female_MedianBehaviorTime.csv')

# Melt the DataFrame to long format
csv_concat = pd.concat([csv3, csv4])
print(csv_concat)
df = csv_concat
# Melt the DataFrame to long format
df_melted = pd.melt(df, id_vars=['Mouse Name', 'Sex'], var_name='Behavior', value_name='Count')

# Sort behaviors alphabetically
df_melted['Behavior'] = pd.Categorical(df_melted['Behavior'], categories=sorted(df_melted['Behavior'].unique()), ordered=True)

# Split the data into male and female
df_male = df_melted[df_melted['Sex'] == 'M'].copy()
df_female = df_melted[df_melted['Sex'] == 'F'].copy()

# Invert the counts for males to place them on the left
df_male['Count'] = -df_male['Count']

# Plot using Seaborn
plt.figure(figsize=(10, 6))
plt.rcParams['pdf.fonttype'] = 42 #to work with in illustrator
plt.rcParams['ps.fonttype'] = 42 #to work with in illustrator

# Create horizontal bar plot for males and females
sns.barplot(data=df_male, y='Behavior', x='Count', color='green', ci='sd', capsize=0.1, orient='h', label='Male')
sns.barplot(data=df_female, y='Behavior', x='Count', color='magenta', ci='sd', capsize=0.1, orient='h', label='Female')

# Individual data points
sns.stripplot(data=df_male, y='Behavior', x='Count', color='black', size=5, alpha=0.7, zorder=1, orient='h')
sns.stripplot(data=df_female, y='Behavior', x='Count', color='black', size=5, alpha=0.7, zorder=1, orient='h')

plt.ylabel('')
plt.xlabel('Total Interactive Behaviors')
plt.title('')

# Remove top and right spines so that only x and y axes are showing
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

# Set the x-axis limits to ensure bars are correctly positioned
max_count = max(df_female['Count'].max(), -df_male['Count'].min())
plt.xlim(-max_count, max_count)

# Define custom legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Female', markerfacecolor='magenta', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Male', markerfacecolor='green', markersize=10)
]

# Create legend and place it in the top left corner
plt.legend(handles=legend_elements, loc='upper right', frameon=False)

# Save file as a PDF
file_path = "/content/MalvsFemaleNumBehaviorsHoriz.pdf"
plt.savefig(file_path)
# files.download(file_path)

# Show the plot
plt.show()