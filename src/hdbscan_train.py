import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stats
import time
from sklearn.manifold import TSNE

def wasserstein(x, y):
    return stats.wasserstein_distance(np.arange(len(x)), np.arange(len(y)), x, y)

def main():
    start_time = time.time()
    
    # Load the data from 1d_power_spectrum_dataset.npz
    print('Loading data...')
    data = np.load('aia171_miniset_pow_spect.npz')
    pow_spect = data['pow_spect']

    # Create and run an instance of HDBSCAN
    print('Running HDBSCAN clustering...')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, gen_min_span_tree=True, metric=wasserstein, core_dist_n_jobs=-1)
    clusterer.fit(pow_spect)
    labels = clusterer.labels_

    # Save the fitted HDBSCAN model
    model_filename = 'hdbscan_model.pkl'
    print(f'Saving the HDBSCAN model to {model_filename}...')
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clusterer, model_file)

    # Perform dimensionality reduction on the power spectrum data
    print('Performing t-SNE dimensionality reduction...')
    embedded_pow_spect = TSNE().fit_transform(pow_spect, metric=wasserstein, n_jobs=-1)

    # Plot the clusters and save the figure as a PNG file
    plt.scatter(embedded_pow_spect[:, 0], embedded_pow_spect[:, 1], c=labels)
    plt.title('HDBSCAN Clustering with EMD Metric')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('hdbscan_clustering.png')

    # Print execution time
    end_time = time.time()
    print(f'Total execution time: {end_time - start_time:.2f} seconds')

if __name__ == "__main__":
    main()