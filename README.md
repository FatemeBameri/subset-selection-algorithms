# subset-selection-algorithms

This repository provides implementations of several advanced subset selection and coreset sampling algorithms, including Determinantal Point Processes (DPP), CLARANS, K-Medoids, Minimum Redundancy Constraint (MRC), Maximum Reduction as Maximum Contribution (MRMC), K-Center Greedy, ELFS, Facility Location, Leverage Score sampling, and KMeans++.
These methods are designed to select diverse, representative, and informative subsets from large datasets. They are particularly useful in machine learning and AI applications where efficient memory usage, scalability, and data diversity are critical — such as anomaly detection, active learning, and large-scale feature memory construction.


## Important Files

### 1. patchcore_coreset.py
The patchcore coreset selects a small, representative subset of samples that preserves the diversity and coverage of the full dataset, ensuring that every original data point is closely represented by at least one selected sample. This approach maximizes the overall similarity between the selected subset and the full dataset, while approximate variants allow efficient scaling to large datasets without sacrificing representativeness.

### 2. Leverage_coreset.py
Leverage Score Sampling is a coreset selection technique that selects a representative subset of data by estimating the importance of each sample with respect to the low-rank structure of the feature space. The method first normalizes the data, then applies a fast randomized Singular Value Decomposition (SVD) to approximate the dominant subspace. Leverage scores are computed as the squared row norms of the left singular vectors, representing how strongly each sample influences the principal subspace. Finally, samples are drawn probabilistically according to these scores, ensuring that highly informative points are more likely to be selected. This approach is particularly effective for large-scale learning tasks where maintaining the geometric structure of the data with a smaller subset is crucial.

### 3. Facility_coreset.py
Facility Location is a submodular coreset selection method that aims to select a diverse and representative subset of samples by maximizing the coverage of the original dataset. The objective is to ensure that each data point in the full dataset is highly similar to at least one selected sample. In our implementation, cosine similarity is used to measure pairwise affinity, and the Apricot library is employed for efficient greedy optimization of the submodular objective. To improve scalability for large datasets, an approximate variant is also used, where the selection is performed on a randomly sampled subset of the data and then mapped back to the original dataset. More details are provided in the paper "Submodular Function Maximization".

### 4. kdpp_coreset.py
The Determinantal Point Process (DPP) is a probabilistic sampling framework designed to select a subset of data with maximum diversity. In other words, it ensures that the chosen items are dissimilar to each other, providing a representative and diverse subset of the dataset. In the discrete case, a kernel matrix 𝐿 is constructed to capture the similarities between all vectors, and the probability of selecting a subset 𝑆 is proportional to the determinant of the corresponding submatrix det; subsets containing similar vectors are therefore less likely to be chosen. Advanced variants, such as k-DPP, allow selecting exactly 𝑘 items, which is particularly useful for representative sampling, coreset selection, data summarization, and active learning. In short, DPP is a method for selecting diverse and non-redundant samples from large datasets.

### 5. Clara_coreset.py
The CLARA algorithm is a scalable sampling-based extension of medoid‑based clustering (such as PAM / k‑medoids), designed to handle large datasets. Instead of clustering the entire data at once — which can be computationally expensive — CLARA draws several small random subsets from the full dataset, runs medoid‑based clustering (e.g. K‑Medoids) separately on each subset to obtain candidate medoids, and then evaluates each candidate set by measuring the total dissimilarity (or cost) between every data point in the full dataset and its nearest medoid. Finally, it selects the medoids (and corresponding clustering) that yield the minimal cost. This way, CLARA approximates the k‑medoids result with much lower computational overhead, making it particularly useful for selecting a representative subset (or “coreset”) of data when dealing with large numbers of data points.

### 6. KMeansPlus_coreset.py
The KMeansPlusSampler is a clustering-based sampling method designed to select a representative subset of data. It first optionally draws a random subset of the full dataset to reduce computation, then applies MiniBatchKMeans with k-means++ initialization to find cluster centers efficiently. The number of clusters is determined by the desired sampling percentage, ensuring that the subset size reflects the proportion of data to be selected. Finally, for each cluster center, the closest actual data point in the full dataset is chosen, producing a subset of samples that effectively represents the distribution and diversity of the original dataset. This approach is particularly useful for coreset selection, data summarization, and speeding up large-scale machine learning tasks.

### 7. MRMC_coreset.py
Implementation of a Maximum Reduction as Maximum Contribution sampler. This is an unofficial implementation inspired by the paper: Efficient Core-set Selection for Deep Learning Through Squared Loss Minimization.

### 8. Improved_Elfs_coreset.py
Our Approximate ELFSCoresetSampler evaluates each feature according to three complementary criteria: representativeness, measured as the distance from the mean of all features; density, estimated approximately via distances to a subset of reference points to capture local concentration; and instability, which quantifies the sensitivity of features to small random perturbations. Each score is normalized and combined into a composite metric, from which the top candidate features are selected. To ensure diversity, a k-center greedy selection is applied on this candidate pool, resulting in a subsample that balances coverage, local density, and robustness to noise, while remaining efficient in both memory usage and computation.

### 9. MRC_coreset.py
Implementation of a Minimum Redundancy Constraint sampler that enables efficient and scalable subset selection in large feature spaces.
This is an unofficial implementation inspired by the paper: Unsupervised surface defect detection using dictionary-based sparse representation.

### 10. Kcenter_coreset.py
Implementation of a K-Center Greedy Sampler for coreset selection. This implementation inspired by the core idea of diversity-based sampling, where samples are iteratively selected to maximize the minimum distance to already chosen points, ensuring wide coverage of the feature space.

### 11. KMedoids.py
Implementation of a K-Medoids Sampler that leverages batch-wise distance computation to efficiently handle large datasets. It preserves the property that medoids are actual data points and safely manages tuple or numpy features, making it suitable for coreset selection in high-dimensional feature spaces.

### 12. Random_coreset.py

### 13. our_run_patchcore.py
This code is based on the run_patchcore.py file from the PatchCore repository and demonstrates how different subset selection methods are called and configured, along with the initialization of their parameters. The design allows for flexibility by offering various subset selection techniques, each with configurable parameters tailored to specific use cases, providing users with adaptable options for their tasks.

---

# supporting-scripts

These scripts provide supporting functionality used throughout the project.

## Files

### 1. train_features_mvtec.py
Feature extraction code for the train set in the MVTec dataset — extracting features from layers 2 and 3 of the ResNet, upsampling layer 3, and then concatenating them.

### 2. test_features_mvtec.py
Feature extraction code for the test set in the MVTec dataset — extracting features from layers 2 and 3 of the ResNet, upsampling layer 3, and then concatenating them. For anomaly images, two separate feature sets should be extracted and saved for each image, containing the normal and anomalous vectors separately.

### 3. mvtec_few_shot_split.py
This script randomly selects a fixed percentage of samples from each class directory and copies them into a new dataset folder. It is intended for few-shot anomaly detection experiments and ensures reproducibility via a fixed random seed.

---

# Related Notes

### 1. Mynotes_ELFS.pdf
The summary notes file of the ELFS paper.

### 2. mynotes_patchcore_minimumredundancy.pdf
The summary notes file for the Minimum Redundancy and PatchCore papers.

### 3. mynotes_CCS.pdf
The summary notes file of the CCS paper.

### 4. subset_methods_results.pdf
The proposed methods are evaluated based on five criteria, including image-level AUROC, pixel-level AUROC, PRO (Per-Region Overlap) score, training time, and inference time, allowing for a thorough comparison of both detection performance and computational efficiency.

## Applications
- Selecting diverse samples for machine learning  
- Data summarization or representative selection  
- Reducing dataset size while maintaining diversity

## Implementation Details
This implementation builds upon the original [PatchCore coreset method](https://github.com/amazon-science/patchcore-inspection/tree/main) and extends it by integrating nine additional coreset-based subset selection strategies. All methods are developed within a unified framework that follows PatchCore’s execution pipeline to ensure fair and consistent evaluation.
