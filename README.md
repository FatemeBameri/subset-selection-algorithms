# subset-selection-algorithms

This repository provides implementations of **subset selection algorithms** based on **Determinantal Point Processes (DPPs)**.  
These methods are useful for selecting diverse and representative subsets of data, especially in machine learning and AI applications.

---

## Files

### 1. `GreedyKDppSampler.py`
Implementation of a **Greedy k-DPP Sampler** that leverages **GPU acceleration** for faster computations.  
It is well-suited for large-scale datasets where efficiency is critical.

### 2. `kDPP_example.py`
A simple example demonstrating how to use **k-DPPs** for subset selection.  
This script is a good starting point to understand the workflow of the algorithm.

### 3. `ApproxCLARANSSampler.py`
Implementation of an **Approximate CLARANS Sampler** that leverages **GPU acceleration** for faster computations.  
It is well-suited for large-scale datasets where efficiency and memory usage are critical.

### 4. `CoverageCentricCoreSampler.py`
Implementation of a **Coverage-Centric Coreset Sampler** that supports **random**, **monotonic**, and **stratified sampling** strategies.

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
