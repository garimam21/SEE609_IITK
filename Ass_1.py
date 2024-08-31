# Given data
Average_Roughness = [0.26,0.31,0.29,0.35,0.21,0.33,0.39,0.29,0.18,0.21]
Film_Thickness= [2.7, 4.2, 5.1, 3.1, 2.5, 2.9, 3.7, 4.1, 2.7, 3.6]
Current = [1.13, 1.51, 1.67, 1.23, 1.38, 1.13, 1.68, 1.57, 1.44, 1.58]

# Function to calculate mean
def mean(data):
    return sum(data) / len(data)

# Function to calculate median
def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    # Calculate Median
    if n % 2 == 0:  # If even number of elements
        mid1 = n // 2
        mid2 = mid1 - 1
        return (sorted_data[mid1] + sorted_data[mid2]) / 2
    else:  # If odd number of elements
        mid = n // 2
        return sorted_data[mid]

# Function to calculate standard deviation
def standard_deviation(data):
    mean_value = mean(data)
    sum_of_squares = 0
    for value in data:
        sum_of_squares += (value - mean_value) ** 2
    variance = sum_of_squares / len(data)
    return variance ** 0.5

# Function to calculate z-score
def calculate_z_scores(data):
    mean_value = mean(data)
    std_dev_value = standard_deviation(data)
    
    z_scores = []
    for x in data:
        z_score = (x - mean_value) / std_dev_value
        z_scores.append(round(z_score, 3))
    return z_scores
z_scores_Film_Thickness = calculate_z_scores(Current)
z_scores_Current = calculate_z_scores(Film_Thickness)
z_scores_Average_Roughness = calculate_z_scores(Average_Roughness)

print("Z-scores For Average_Roughness:", calculate_z_scores(Average_Roughness))
print("Z-scores For Film_Thickness:",calculate_z_scores(Film_Thickness) )
print("Z-scores For Current:",calculate_z_scores(Current))

# Function to calculate Kurtosis
def Kurtosis(data):
    n = len(data)
    avg = mean(data)
    std_dev_value = standard_deviation(data)
    
    sum = 0
    for x in data:
        z = (x - avg)
        sum += z ** 4
    sum = sum/(std_dev_value)**4   
    
    Kurtosis = (n * (n + 1) * sum) / ((n - 1) * (n - 2) * (n - 3)) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    
    return Kurtosis

print("Kurtosis for Average_Roughness:", Kurtosis( Average_Roughness))
print("Kurtosis for Film_Thickness:", Kurtosis( Film_Thickness ) )
print("Kurtosis for Current:", Kurtosis( Current ) )

import matplotlib.pyplot as plt

# Create a scatter plots for thickness vs roughness,
#thickness vs current, and roughness vs current.

plt.figure(figsize=(8, 6))
plt.scatter(Film_Thickness,Current, color='blue', marker='o')
plt.title('Scatter Plot of Film Thickness vs. Current')
plt.xlabel('Film_Thickness')
plt.ylabel('Current')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(Average_Roughness,Current, color='red', marker='o')
plt.title('Scatter Plot of Average_Roughness vs. Current')
plt.xlabel('Average_Roughness')
plt.ylabel('Current')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(Average_Roughness,Film_Thickness, color='yellow', marker='o')
plt.title('Scatter Plot of Average_Roughness vs. Film_Thickness')
plt.xlabel('Average_Roughness')
plt.ylabel('Film_Thickness')
plt.grid(True)
plt.show()

# Create scatter plots for z-score for thickness vs roughness,
#thickness vs current, and roughness vs current

plt.figure(figsize=(8, 6))
plt.scatter(z_scores_Film_Thickness, z_scores_Current, color='green', marker='o')
plt.title('Scatter Plot of Z-Scores for Film Thickness vs. Current')
plt.xlabel('Z-Scores of Film Thickness')
plt.ylabel('Z-Scores of Current')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(z_scores_Average_Roughness, z_scores_Current, color='violet', marker='o')
plt.title('Scatter Plot of Z-Scores for Film Thickness vs. Current')
plt.xlabel('Z_scores_Average_Roughness')
plt.ylabel('Z-Scores of Current')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(z_scores_Average_Roughness, z_scores_Film_Thickness, color='pink', marker='o')
plt.title('Scatter Plot of Z-Scores for Film Thickness vs. Current')
plt.xlabel('Z_scores_Average_Roughness')
plt.ylabel('Z_scores_Film_Thickness')
plt.grid(True)
plt.show()

# Function to calculate covariance
def covariance(x, y):
    n = len(x)
    avgx = mean(x)
    avgy = mean(y)
    std_dev_value_x = standard_deviation(x)
    std_dev_value_y = standard_deviation(y)
    summation = 0
    sum_cov = 0
    for xi, yi in zip(x, y):
        sum_cov += (xi - avgx) * (yi - avgy)
    return sum_cov / (n - 1)

 #Function to calculate correlation
def correlation(x, y):
    cov_xy = covariance(x, y)
    std_x = standard_deviation(x)
    std_y = standard_deviation(y)
    return cov_xy / (std_x * std_y)

print("Covariance between Average_Roughness and Film_Thickness:", round(covariance(Average_Roughness, Film_Thickness), 3))
print("Covariance between Average_Roughness and Current:", round(covariance(Average_Roughness, Current), 3))
print("Covariance between Film_Thickness and Current:", round(covariance(Film_Thickness, Current), 3))

print("Correlation between Average_Roughness and Film_Thickness:", round(correlation(Average_Roughness, Film_Thickness), 3))
print("Correlation between Average_Roughness and Current:", round(correlation(Average_Roughness, Current), 3))
print("Correlation between Film_Thickness and Current:", round(correlation(Film_Thickness, Current), 3))


# QUESTION= 3
# Given data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Data Preparation
#GDP per capita: 
GPC = [7.695, 9.342, 9.853, 6.635, 7.364, 9.557, 9.054, 10.873, 8.755, 10.817, 11.488, 11.023, 8.145]
#social Support:
SS=[0.463, 0.802, 0.91, 0.49, 0.619, 0.847, 0.762, 0.903, 0.603, 0.843, 0.915, 0.92, 0.708]
#Healthy_Life_Expectancy: 
HLE =[52.493, 66.005, 66.253, 53.4, 48.478, 68.001, 66.402, 72.5, 60.633, 66.9, 76.953, 68.2, 55.809]



# Step 2: Standardize the Data
mean_GPC = mean(GPC)
mean_SS = mean(SS)
mean_HLE = mean(HLE)
std_GPC=standard_deviation(GPC)
std_SS=standard_deviation(SS)
std_HLE=standard_deviation(HLE)
for x in GPC:
    GPC_standardized = [(x - mean_GPC) / std_GPC ]
for y in SS:
    SS_standardized = [(y - mean_SS) / std_SS ]
for z in GPC:
    HLE_standardized = [(z - mean_HLE) / std_HLE]


# Combining standardized data into a matrix
data_standardized = np.array([GPC_standardized, SS_standardized, HLE_standardized]).T 
print("Standardized Data:\n", data_standardized)

# Given Eigenvectors
eigenvector1 = np.array([0.584, 0.795, 0.162])
eigenvector2 = np.array([0.571, -0.261, -0.778])
eigenvector3 = np.array([0.576, -0.547, 0.607])

# Step 2: Project the data onto each eigenvector (PCA1, PCA2, PCA3)
PCA1 = np.dot(data_standardized, eigenvector1)
PCA2 = np.dot(data_standardized, eigenvector2)
PCA3 = np.dot(data_standardized, eigenvector3)

# Output the projections
print("Projections on PCA1:\n",PCA1)
print("Projections on PCA2:\n",PCA2)
print("Projections on PCA3:\n",PCA3)



