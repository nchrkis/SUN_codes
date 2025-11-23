'''
Generic version of the SUN GMM-refinement algorithm
developed by Nicholas Christakis and Dimitris Drikakis

Flowchart of algorithm:
A CSV datafile and a set of dominant cluster centres (from RUN-ICON)
are input.
Data is read, converted to a numpy array and normalised using min–max
scaling.
The Gaussian Mixture Model (GMM) is initialised using the RUN-ICON
centres and fitted through the Expectation–Maximisation (EM) procedure.
Soft membership probabilities, refined centres, covariance matrices,
and updated cluster assignments are obtained.
Probability ranges are evaluated around each cluster centre and at the
cluster edges using Mahalanobis-distance-based radii.
Refined centres, cluster sizes, probability ranges, and denormalised
cluster data are written to output files for further post-processing
and visualisation.

PLEASE NOTE:
This algorithm is intended to be used *after* RUN-ICON has determined
the optimal number of clusters and produced dominant, stable centres.
The GMM step provides probabilistic refinement, noise reduction, and
uncertainty quantification for each data point.

REFERENCE:
If you use this software in any publication, please cite:

Christakis, N.; Drikakis, D.
"SUN: Stochastic UNsupervised learning for data noise and uncertainty
reduction", Submitted to Applied Sciences (2025).

DISCLAIMER:
This software is provided “as is” without warranty of any kind.
The author(s) shall not be held responsible for any damages, losses,
or consequences resulting from the use or misuse of this software.
Users are responsible for ensuring compliance with all applicable laws.
By using this software, you acknowledge and agree to use it entirely
at your own risk.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from numpy import loadtxt

# Number of clusters required
No_clusters = 3  # Change this to the desired number of clusters

# Read the centers that have resulted from RUN-ICON from a text file
with open("most_common_centroid.txt", "r") as file:
    centers = [list(map(float, line.strip().split(","))) for line in file]
centers = np.array(centers)

# Read input features
filename = "your_file.csv"
features = pd.read_csv(filename, header=None) #assuming there is no header in the file
#drop lines with empty values
df = df.dropna()

#Convert to numpy array
feature = features.to_numpy()

#Normalize using min-max
feat_norm = (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))


# Utilize KMeans from sklearn with given centers
kmeans = KMeans(n_clusters=No_clusters, init=centers, n_init=1)
kmeans.fit(feat_norm)
y_km = kmeans.predict(feat_norm)

# Denormalize if applicable for plotting purposes
centroid_denorm = kmeans.cluster_centers_ * (feature.max(axis=0) - feature.min(axis=0)) + feature.min(axis=0)
feat_denorm = feat_norm * (feature.max(axis=0) - feature.min(axis=0)) + feature.min(axis=0)

# Condition to plot only if dimensions are <= 2
if feature.shape[1] <= 2:
    plt.figure(figsize=(16, 14))
    plt.scatter(feat_denorm[:, 0], feat_denorm[:, 1], c=y_km, s=40, cmap='rainbow', alpha=0.6, label='RUN-ICON clusters')
    plt.xticks(fontsize=35) # Increase the font size of the numbers on the x-axis
    plt.yticks(fontsize=35) # Increase the font size of the numbers on the y-axis
    plt.tick_params(axis='both', which='both', width=2, direction='out')
    plt.tick_params(axis='both', which='major', length=6)
    plt.tick_params(axis='both', which='minor', length=4)
    plt.gca().spines['bottom'].set_linewidth(4)  # set thickness of x-axis
    plt.gca().spines['left'].set_linewidth(4)    # set thickness of y-axis
    # Remove the plot frame
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('X', fontsize=35)
    plt.ylabel('Y', fontsize=35)
    plt.savefig('Clusters_KMeans.jpg')
    plt.show()

# Fit Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=No_clusters, init_params='kmeans', means_init=kmeans.cluster_centers_)
gmm.fit(feat_norm)
y_gmm = gmm.predict(feat_norm)

##########################Get probabilities around centres and edges of clusters based on distance###################################
##########################and a radius around centres####################################################
# --- Compute membership probabilities for all points ---
probs = gmm.predict_proba(feat_norm)   # shape: (N_points, No_clusters)


center_ranges = {}
edge_ranges = {}
cluster_counts = {}

n_features = feat_norm.shape[1]
reg_covar = getattr(gmm, "reg_covar", 1e-6)  # small regularizer (works if gmm has attribute)

for k in range(No_clusters):
    # global boolean mask of points assigned to cluster k
    in_cluster = (y_gmm == k)
    cluster_counts[k] = int(in_cluster.sum())   # number of points assigned to cluster k

    # get the actual points assigned to cluster k
    cluster_points = feat_norm[in_cluster]
    cluster_probs  = probs[in_cluster, k]   # probabilities (for cluster k) for those points

    # handle empty cluster
    if cluster_points.shape[0] == 0:
        center_ranges[k] = (None, None)
        edge_ranges[k]   = (None, None)
        continue

    # cluster mean (we use GMM mean to be consistent, but one can also use cluster_points.mean(axis=0))
    mu = gmm.means_[k]

    # get covariance for this component and build invertible matrix
    cov = gmm.covariances_[k]
    # If covariance_type is 'diag', cov will be 1-D per component
    if cov.ndim == 1:
        cov_mat = np.diag(cov + reg_covar)
    else:
        cov_mat = cov + np.eye(n_features) * reg_covar

    # invert covariance (use pseudo-inverse if necessary)
    try:
        cov_inv = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_mat)

    # Mahalanobis distances for points in this cluster (local distances)
    diff = cluster_points - mu  # shape = (n_cluster_points, n_features)
    # efficient Mahalanobis: sqrt( (diff @ cov_inv) * diff ) sum across features
    m_sq = np.sum((diff @ cov_inv) * diff, axis=1)   # squared Mahalanobis
    m_dist = np.sqrt(np.maximum(m_sq, 0.0))

    # define radius using Mahalanobis (max of distances in this cluster)
    radius_k = m_dist.max()

    # We may change 0.5 to any fraction we prefer
    centre_local = m_dist <= 0.5 * radius_k
    edge_local   = m_dist >  0.5 * radius_k

    # compute min/max probabilities for centre and edge (within cluster)
    if centre_local.any():
        cmin = float(cluster_probs[centre_local].min())
        cmax = float(cluster_probs[centre_local].max())
        center_ranges[k] = (cmin, cmax)
    else:
        center_ranges[k] = (None, None)

    if edge_local.any():
        emin = float(cluster_probs[edge_local].min())
        emax = float(cluster_probs[edge_local].max())
        edge_ranges[k] = (emin, emax)
    else:
        edge_ranges[k] = (None, None)

# Print summary: counts + probability ranges
for k in range(No_clusters):
    print(f"\nCluster {k+1}:")
    print(f"  Count = {cluster_counts[k]}")
    cmn, cmx = center_ranges[k]
    emn, emx = edge_ranges[k]
    print(f"  Centre prob range: {cmn if cmn is not None else 'None'} -> {cmx if cmx is not None else 'None'}")
    print(f"  Edge   prob range: {emn if emn is not None else 'None'} -> {emx if emx is not None else 'None'}")
######################################################################################################################

# Output the centers and the number of particles around each cluster center
cluster_centers = gmm.means_
unique_labels, counts = np.unique(y_gmm, return_counts=True)
with open('SUN_cluster_info.txt', 'w') as file:
    for label, count in zip(unique_labels, counts):
        file.write(f"Cluster {label+1}: {count} instances\n")
        file.write(f"Center: {cluster_centers[label]}\n")

# Save denormalized cluster data to separate CSV files
for i in range(No_clusters):
    cluster_data = feat_denorm[y_gmm == i]
    cluster_df = pd.DataFrame(cluster_data)
    cluster_df.to_csv(f'cluster_{i+1}.csv', index=False)

# Plot GMM clusters if dimensions are <= 2
if feature.shape[1] <= 2:
    plt.figure(figsize=(16, 14))
    plt.scatter(feat_denorm[:, 0], feat_denorm[:, 1], c=y_gmm, s=40, cmap='rainbow', alpha=0.6, label='SUN clusters')
    plt.xticks(fontsize=35) # Increase the font size of the numbers on the x-axis
    plt.yticks(fontsize=35) # Increase the font size of the numbers on the y-axis
    plt.tick_params(axis='both', which='both', width=2, direction='out')
    plt.tick_params(axis='both', which='major', length=6)
    plt.tick_params(axis='both', which='minor', length=4)
    plt.gca().spines['bottom'].set_linewidth(4)  # set thickness of x-axis
    plt.gca().spines['left'].set_linewidth(4)    # set thickness of y-axis
    # Remove the plot frame
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('X', fontsize=35)
    plt.ylabel('Y', fontsize=35)
    plt.savefig('Clusters_SUN.jpg')
    plt.legend(fontsize=15)
    plt.show()
