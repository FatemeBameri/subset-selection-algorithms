# subset-selection-algorithms

This repository provides implementations of **subset selection algorithms** based on **Determinantal Point Processes (DPPs)**.  
These methods are useful for selecting diverse and representative subsets of data, especially in machine learning and AI applications.

---

## Files

### 1. `GreedyKDppSampler.py`
Implementation of a **Greedy k-DPP Sampler** that leverages **GPU acceleration** for faster computations.  
It is well-suited for large-scale datasets where efficiency is critical.

### 2. `ApproxCLARANSSampler.py`
Implementation of an **Approximate CLARANS Sampler** that leverages **GPU acceleration** for faster computations.  
It is well-suited for large-scale datasets where efficiency and memory usage are critical.

### 3. `CoverageCentricCoreSampler.py`
Implementation of a **Coverage-Centric Coreset Sampler** that supports **random**, **monotonic**, and **stratified sampling** strategies.  
This implementation is **adapted from the original code** available on GitHub: [Coverage-Centric Coreset](https://github.com/haizhongzheng/Coverage-centric-coreset-selection/tree/main) and has been integrated into this project.

---

## Applications
- Selecting diverse samples for machine learning  
- Data summarization or representative selection  
- Reducing dataset size while maintaining diversity  

---

## Usage

Run the basic example:
```bash
python kDPP_example.py
