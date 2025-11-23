'''
Generic version of the RUN-ICON algorithm
developed by Nicholas Christakis and Dimitris Drikakis

Flowchart of algorithm:
A CSV datafile is input
Data is read, saved in a numpy array and normalized, based on the max-min values of each column
Within RUN-ICON, for every loop, the Generalized Centre Coordinate is calculated for comparisons 
Frequency of occurrence of most dominant centres is calculated
Centres are output in a file for future use
PLEASE NOTE: If you want to test frequency of occurrence of dominant centres for different number of clusters 
              you must modify variable "i_cluster" manually in the code
 

DISCLAIMER:
This software is provided "as is" without warranty of any kind. 
The author(s) shall not be held responsible for any damages, losses, or consequences resulting from the use or misuse of this software. 
Users are encouraged to ensure the software is used for its intended purpose and in compliance with applicable laws.
By using this software, you acknowledge and agree to use it entirely at your own risk.

If you use this software in any publication, please cite the following paper:
Christakis, N.; Drikakis, D. "Reducing Uncertainty and Increasing Confidence in Unsupervised Learning." Mathematics 2023, 11, 3063. https://doi.org/10.3390/math11143063
'''

from numpy import array, vstack, loadtxt
from scipy.cluster.vq import kmeans2
import pandas as pd
import numpy as np
import time

# Start the timer
start_time = time.time()

#read data and get rid of NaNs 
file_path = 'your_file.csv'  
df = pd.read_csv(file_path, header=None) #assuming there is no header in the file
#drop lines with empty values1
df = df.dropna()

#Convert to numpy array
feature = df.to_numpy()



#Normalize all columns between 0 and 1 with (x-x_min)/(x_max-x_min)
feat_norm = (feature - feature.min(axis=0)) / (feature.max(axis=0)-feature.min(axis=0))


####################### Various parameters needed for the model########################################
i_cluster = 3  # Number of clusters examined for finding dominant cluster centres
	           #
	           #This number should be changed manually by the user
	           #to examine the frequency of occurrence of dominant clusters for this number
	           #
	           #i_cluster is recommended to start from values greater than 2, since for 2 clusters, 
	           #the probability of finding dominant centres is extremely high
	           #
tol = 1.e-1    # Tolerance when calculating proximity of centres
               #
               #Suggested to be 10^-2 if your data is more than 1,000 lines
               #otherwise, 10^-1 will suffice
               #
TOT_N = 10     #Total loops of algorithm application to find dominant centres
               #  
meanv = 0      #counter for calculation of mean CONfidence from  all algorithm application


# Detect the number of features dynamically
num_features = feature.shape[1]

for K_stat in range(TOT_N):  
	N = 100        #number of k-means++ applications to find how many times
	               #out of N dominant centre appears

    # Perform K-means++ clustering
	centroid, label = kmeans2(feat_norm, i_cluster, iter=100, minit='++')
	centr = array([centroid[:, i] for i in range(num_features)]).T

	for i in range(1, N):
		centroid, label = kmeans2(feat_norm, i_cluster, iter=100, minit='++')
		centr = vstack((centr, array([centroid[:, i] for i in range(num_features)]).T))

    # Reshape the array to group cluster center coordinates
	centr = np.reshape(centr, (N, i_cluster, num_features))

    # Gemeralized centre coordinate
    #Sum of centers for comparison
	centr_sum = np.sum(centr[0], axis=0)
	for i in range(1, N):
		centr_sum_line = np.sum(centr[i], axis=0)
		centr_sum = vstack((centr_sum, centr_sum_line))

	clust = np.zeros((N, 2), dtype=np.int32)
	clust_bool = np.zeros(N, dtype=bool)
	i_c = 0
	centr_sum1 = centr_sum.copy()

	for i in centr_sum:
		cl_no = 1
		i_iter = 0
		for j in centr_sum1:
			if clust_bool[i_iter]:
				i_iter += 1
				continue
			if np.all(np.abs(i - j) < tol):
				cl_no += 1
				clust_bool[i_iter] = True
			i_iter += 1
		cl_no -= 1
		clust[i_c, 0] = cl_no
		clust[i_c, 1] = i_c
		i_c += 1

		if np.sum(clust[:, 0]) >= 100:
			break

	print(clust[:, 0].max(axis=0))
	meanv += clust[:, 0].max(axis=0)


# Retrieve the most frequent centroid
most_common_centroid_index = np.argmax(clust[:, 0])
most_common_centroid_position = clust[most_common_centroid_index, 1]
most_common_centroid = centr[most_common_centroid_position]

# Save the most common centroid in a new file
output_file = 'most_common_centroid.txt'
np.savetxt(output_file, most_common_centroid, fmt='%f', delimiter=',', comments='')

print(f"Centroid with the highest occurrence frequency saved to {output_file}.")


# Assign all points to nearest dominant centroid
from scipy.spatial.distance import cdist
distances = cdist(feat_norm, most_common_centroid)
labels_final = np.argmin(distances, axis=1)
# Count points per cluster (true integer counts)
unique, counts = np.unique(labels_final, return_counts=True)
cluster_sizes_final = dict(zip(unique, counts))
print("\nCluster sizes based on final dominant centroids (ICON):")
for k, v in cluster_sizes_final.items():
    print(f"  Cluster {k}: {v} points")
#Save number of points in a file
np.savetxt("final_RUN_ICON_cluster_sizes.txt", np.array(list(cluster_sizes_final.values())),
           fmt="%d", header="Cluster sizes based on dominant centroids (ICON)")



#Denormalize most common centroids 
original_max = feature.max(axis=0)
original_min = feature.min(axis=0)
denormalized_centroid = original_min + (most_common_centroid * (original_max - original_min))


# Save the denormalized centroid to a text file
denormalized_output_file = 'denormalized_most_common_centroid.txt'
np.savetxt(denormalized_output_file, denormalized_centroid, fmt='%.3f', delimiter=',', comments='')

print(f"\nDenormalized centroids saved to {denormalized_output_file}.\n")
#################################################################################


meanv = meanv / TOT_N
print('Mean value = ', meanv)
'''
The user is advised to plot the mean value vs the number of clusters with error bars for the maximum
and minimum values of dominant centres occurrence
High  mean value corresponds to Incresed CONfidence (ICON)
Low diference between maximum and minimum values corresponds to Reduced UNcertainty (RUN)
'''

# End the timer and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"For {i_cluster} clusters. Elapsed time: {elapsed_time:.4f} seconds.")
