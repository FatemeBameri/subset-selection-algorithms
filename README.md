# subset-selection-algorithms

This repository provides implementations of **subset selection algorithms** based on **Determinantal Point Processes, CLARANS Sampler, Coverage-Centric Coreset Sampler, K-Medoids Sampler, Minimum Redundancy Constraint sampler, Maximum Reduction as Maximum Contribution sampler**.  
These methods are useful for selecting diverse and representative subsets of data, especially in machine learning and AI applications.

---

## Files

### 1. `KDpp.py`
Implementation of a **Greedy k-DPP Sampler** that leverages **GPU acceleration** for faster computations.  
It is well-suited for large-scale datasets where efficiency is critical.

### 2. `CLARANS.py`
Implementation of an **Approximate CLARANS Sampler** that leverages **GPU acceleration** for faster computations.  
It is well-suited for large-scale datasets where efficiency and memory usage are critical.

### 3. `CCC.py`
Implementation of a **Coverage-Centric Coreset Sampler** that supports **random**, **monotonic**, and **stratified sampling** strategies.  
This implementation is **adapted from the original code** available on GitHub: [Coverage-Centric Coreset](https://github.com/haizhongzheng/Coverage-centric-coreset-selection/tree/main) and has been integrated into this project.

### 4. `KMedoids.py`
Implementation of a **K-Medoids Sampler** that leverages **GPU acceleration** and **batch-wise distance computation** to efficiently handle large datasets.  
It preserves the property that **medoids are actual data points** and safely manages **tuple or numpy features**, making it suitable for coreset selection in high-dimensional feature spaces.

### 5. `MRC.py`
Implementation of a **Minimum Redundancy Constraint sampler** that leverages **GPU acceleration** to enable efficient and scalable subset selection in large feature spaces.
This is an unofficial implementation inspired by the paper: **Unsupervised surface defect detection using dictionary-based sparse representation**.

### 6. `MRMC.py`
Implementation of a **Maximum Reduction as Maximum Contribution sampler**. This is an unofficial implementation inspired by the paper: **Efficient Core-set Selection for Deep Learning Through Squared Loss Minimization.**

---

## Applications
- Selecting diverse samples for machine learning  
- Data summarization or representative selection  
- Reducing dataset size while maintaining diversity
  
