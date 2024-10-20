# -*- coding: utf-8 -*-
"""SimBATOIGroundTruthStats.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OZTujorBo8Ipf3VpNfH3PvOod0UGLH4J
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, mannwhitneyu, sem, t
from math import sqrt

# Your provided dataset
data = {
    'Behavior': ['Paired Exploration', 'Paired Exploration', 'Paired Exploration', 'Paired Exploration',
                 'Paired Exploration', 'Paired Exploration', 'Paired Exploration', 'Paired Exploration',
                 'Pursuit', 'Pursuit', 'Pursuit', 'Pursuit',
                 'Pursuit', 'Pursuit', 'Pursuit', 'Pursuit',
                 'Anogenital Sniffing', 'Anogenital Sniffing', 'Anogenital Sniffing', 'Anogenital Sniffing',
                 'Anogenital Sniffing', 'Anogenital Sniffing', 'Anogenital Sniffing', 'Anogenital Sniffing',
                 'Nose Nose', 'Nose Nose', 'Nose Nose', 'Nose Nose',
                 'Nose Nose', 'Nose Nose', 'Nose Nose', 'Nose Nose',
                 'Side Side', 'Side Side', 'Side Side', 'Side Side',
                 'Side Side', 'Side Side', 'Side Side', 'Side Side'],
    'Method': ['SimBA', 'SimBA', 'SimBA', 'SimBA', 'Manual', 'Manual', 'Manual', 'Manual',
               'SimBA', 'SimBA', 'SimBA', 'SimBA', 'Manual', 'Manual', 'Manual', 'Manual',
               'SimBA', 'SimBA', 'SimBA', 'SimBA', 'Manual', 'Manual', 'Manual', 'Manual',
               'SimBA', 'SimBA', 'SimBA', 'SimBA', 'Manual', 'Manual', 'Manual', 'Manual',
               'SimBA', 'SimBA', 'SimBA', 'SimBA', 'Manual', 'Manual', 'Manual', 'Manual'],
    'Bouts': [29, 21, 30, 20, 28, 23, 28, 26, 23, 13, 38, 19, 17, 12, 10, 9, 8, 5, 1, 4, 9, 5, 4, 4, 39, 27, 18, 16, 33, 17, 14, 18, 27, 33, 37, 21, 36, 35, 34, 25]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Function to calculate Cohen's d for paired samples
def cohens_d_paired(group1, group2):
    """Calculate Cohen's d for paired samples.

    Parameters:
    group1 (array-like): Data for group 1.
    group2 (array-like): Data for group 2.

    Returns:
    float: Cohen's d effect size.
    """
    diff = group1 - group2
    return np.mean(diff) / np.std(diff, ddof=1)

# Loop through each behavior
behaviors = df['Behavior'].unique()
results = []

for behavior in behaviors:
    # Filter data for the current behavior
    df_behavior = df[df['Behavior'] == behavior]

    # Split data based on method
    simba = df_behavior[df_behavior['Method'] == 'SimBA']['Bouts'].values
    manual = df_behavior[df_behavior['Method'] == 'Manual']['Bouts'].values

    # Perform Shapiro-Wilk test on the differences
    diff = simba - manual
    statistic, p_value = shapiro(diff)
    print(f"Shapiro-Wilk p-value for {behavior}: \t", p_value)

    if p_value > 0.05:
        t_statistic, p_value = ttest_rel(simba, manual)
        test_used = "Paired t-test"
    else:
        t_statistic, p_value = mannwhitneyu(simba, manual)
        test_used = "Mann-Whitney U Test"

    # Calculate Cohen's d for paired samples
    cohens_d_value = cohens_d_paired(simba, manual)

    # Confidence Interval for the differences
    mean_diff = np.mean(diff)
    sem_diff = sem(diff)
    degrees_freedom = len(diff) - 1

    # Calculating the 95% confidence interval
    confidence_interval = t.interval(0.95, degrees_freedom, loc=mean_diff, scale=sem_diff)

    # Store results
    results.append({
        'Behavior': behavior,
        'Test Used': test_used,
        'T-statistic': t_statistic,
        'P-value': p_value,
        'Cohens d': cohens_d_value,
        '95% CI': confidence_interval
    })

# Print the results
for result in results:
    print(f"\nBehavior: {result['Behavior']}")
    print(f"Test Used: {result['Test Used']}")
    print(f"T-statistic:\t {result['T-statistic']}")
    print(f"P-value:\t {result['P-value']}")
    print(f"Cohens d:\t {result['Cohens d']}")
    print(f"95% CI:\t {result['95% CI']}")

import numpy as np
from scipy import stats

# Data
hand = np.array([120, 66, 75, 148])
toi = np.array([129, 101, 115, 140])

# Calculate differences
differences = hand - toi

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(hand, toi)

# Calculate mean and standard deviation of differences
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)

# Calculate Cohen's d
cohen_d = mean_diff / std_diff

# Calculate 95% confidence interval
conf_int = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=std_diff/np.sqrt(len(differences)))

# Output results
print(f"Mean difference: {mean_diff}")
print(f"Standard deviation of differences: {std_diff}")
print(f"Cohen's d: {cohen_d}")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print(f"95% Confidence Interval for the mean difference: {conf_int}")

import numpy as np
from scipy import stats

# Data
hand = np.array([120, 66, 75, 148])
toi = np.array([129, 101, 115, 140])

# Calculate differences
differences = hand - toi

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(hand, toi)

# Calculate mean and standard deviation of differences
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)

# Calculate Cohen's d
cohen_d = mean_diff / std_diff

# Calculate 95% confidence interval
conf_int = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=std_diff/np.sqrt(len(differences)))

# Output results
print(f"Mean difference: {mean_diff}")
print(f"Standard deviation of differences: {std_diff}")
print(f"Cohen's d: {cohen_d}")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print(f"95% Confidence Interval for the mean difference: {conf_int}")

import numpy as np
from scipy import stats

# Data
hand = np.array([14, 10, 7, 5])
toi = np.array([11, 6, 4, 5])


# Calculate differences
differences = hand - toi

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(hand, toi)

# Calculate mean and standard deviation of differences
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)

# Calculate Cohen's d
cohen_d = mean_diff / std_diff

# Calculate 95% confidence interval
conf_int = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=std_diff/np.sqrt(len(differences)))

# Output results
print(f"Mean difference: {mean_diff}")
print(f"Standard deviation of differences: {std_diff}")
print(f"Cohen's d: {cohen_d}")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print(f"95% Confidence Interval for the mean difference: {conf_int}")