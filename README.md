# subset-selection-algorithms

This repository provides implementations of **subset selection algorithms** based on **Determinantal Point Processes sampler, CLARANS sampler, K-Medoids sampler, Minimum Redundancy Constraint sampler, Maximum Reduction as Maximum Contribution sampler, KCenterGreedy sampler, ELFS sampler, Facility Location sampler, Leverage sampler, Kmeans++ sampler**.  
These methods are useful for selecting diverse and representative subsets of data, especially in machine learning and AI applications.

---

## Important Notes on Files

### 1. `LeverageCoreset.py`
Leverage Score Sampling is a coreset selection technique that selects a representative subset of data by estimating the importance of each sample with respect to the low-rank structure of the feature space. The method first normalizes the data, then applies a fast randomized Singular Value Decomposition (SVD) to approximate the dominant subspace. Leverage scores are computed as the squared row norms of the left singular vectors, representing how strongly each sample influences the principal subspace. Finally, samples are drawn probabilistically according to these scores, ensuring that highly informative points are more likely to be selected. This approach is particularly effective for large-scale learning tasks where maintaining the geometric structure of the data with a smaller subset is crucial.

### 2. `FacilityLocation.py`
Facility Location is a submodular coreset selection method that aims to select a diverse and representative subset of samples by maximizing the coverage of the original dataset. The objective is to ensure that each data point in the full dataset is highly similar to at least one selected sample. In our implementation, cosine similarity is used to measure pairwise affinity, and the Apricot library is employed for efficient greedy optimization of the submodular objective. To improve scalability for large datasets, an approximate variant is also used, where the selection is performed on a randomly sampled subset of the data and then mapped back to the original dataset. More details are provided in the paper "Submodular Function Maximization".

### 3. `kdppCoreset.py`
The Determinantal Point Process (DPP) is a probabilistic sampling framework designed to select a subset of data with maximum diversity. In other words, it ensures that the chosen items are dissimilar to each other, providing a representative and diverse subset of the dataset. In the discrete case, a kernel matrix 𝐿 is constructed to capture the similarities between all vectors, and the probability of selecting a subset 𝑆 is proportional to the determinant of the corresponding submatrix det; subsets containing similar vectors are therefore less likely to be chosen. Advanced variants, such as k-DPP, allow selecting exactly 𝑘 items, which is particularly useful for representative sampling, coreset selection, data summarization, and active learning. In short, DPP is a method for selecting diverse and non-redundant samples from large datasets.

### 4. `CLARACoreset.py`
The CLARA algorithm is a scalable sampling-based extension of medoid‑based clustering (such as PAM / k‑medoids), designed to handle large datasets. Instead of clustering the entire data at once — which can be computationally expensive — CLARA draws several small random subsets from the full dataset, runs medoid‑based clustering (e.g. K‑Medoids) separately on each subset to obtain candidate medoids, and then evaluates each candidate set by measuring the total dissimilarity (or cost) between every data point in the full dataset and its nearest medoid. Finally, it selects the medoids (and corresponding clustering) that yield the minimal cost. This way, CLARA approximates the k‑medoids result with much lower computational overhead, making it particularly useful for selecting a representative subset (or “coreset”) of data when dealing with large numbers of data points.

### 5. `KMeansPlusCoreset.py*********`
The KMeansPlusSampler is a clustering-based sampling method designed to select a representative subset of data. It first optionally draws a random subset of the full dataset to reduce computation, then applies MiniBatchKMeans with k-means++ initialization to find cluster centers efficiently. The number of clusters is determined by the desired sampling percentage, ensuring that the subset size reflects the proportion of data to be selected. Finally, for each cluster center, the closest actual data point in the full dataset is chosen, producing a subset of samples that effectively represents the distribution and diversity of the original dataset. This approach is particularly useful for coreset selection, data summarization, and speeding up large-scale machine learning tasks.

### 6. `MRMC.py`
Implementation of a Maximum Reduction as Maximum Contribution sampler. This is an unofficial implementation inspired by the paper: Efficient Core-set Selection for Deep Learning Through Squared Loss Minimization.

### 7. `ELFSCoreset.py*************`


### 8. `MRC.py`
Implementation of a Minimum Redundancy Constraint sampler that enableس efficient and scalable subset selection in large feature spaces.
This is an unofficial implementation inspired by the paper: Unsupervised surface defect detection using dictionary-based sparse representation.

### 9. `KCenterGreedySampler.py`
Implementation of a K-Center Greedy Sampler for coreset selection. This implementation inspired by the core idea of diversity-based sampling, where samples are iteratively selected to maximize the minimum distance to already chosen points, ensuring wide coverage of the feature space.

### 10. `KMedoids.py`
Implementation of a K-Medoids Sampler that leverages batch-wise distance computation to efficiently handle large datasets.  
It preserves the property that medoids are actual data points and safely manages tuple or numpy features, making it suitable for coreset selection in high-dimensional feature spaces.

---

# supporting-scripts

These scripts provide supporting functionality used throughout the project.

## Files

### 1. `train_features_mvtec.py`
Feature extraction code for the train set in the MVTec dataset — extracting features from layers 2 and 3 of the ResNet, upsampling layer 3, and then concatenating them.

### 2. `test_features_mvtec.py`
Feature extraction code for the test set in the MVTec dataset — extracting features from layers 2 and 3 of the ResNet, upsampling layer 3, and then concatenating them. For anomaly images, two separate feature sets should be extracted and saved for each image, containing the normal and anomalous vectors separately.

---

# Related Notes

### 1. `Mynotes_ELFS.pdf`
The summary notes file of the ELFS paper.

### 2. `mynotes_patchcore_minimumredundancy.pdf`
The summary notes file for the Minimum Redundancy and PatchCore papers.

### 3. `mynotes_CCS.pdf`
The summary notes file of the CCS paper.

## Applications
- Selecting diverse samples for machine learning  
- Data summarization or representative selection  
- Reducing dataset size while maintaining diversity
  
