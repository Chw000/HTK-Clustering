# A Hybrid Clustering Method Based On K-means Algorithm

This repository is the official implementation of [A Hybrid Clustering Method Based On K-means Algorithm]. 


## Introduction

We propose a novel hybrid algorithm that effectively combines K-means clustering and hierarchical and uses triangle inequality to accelerate the clustering speed. The HTK clustering algorithm can produce the same clustering results as the standard K-means clustering algorithm. The proposed algorithm is superior to the standard K-means clustering algorithm in terms of running time and memory usage, thus improving the clustering speed and time complexity of the algorithm. The proposed clustering methods are tested on sci-kit learn datasets, and they are more favorable than the random restart k-means algorithm.


## Datasets
You can download datasets here:


### Real Datasets
- [Real_Datasets](https://github.com/Chw000/A-Hybrid-Clustering-Method-Based-On-K-means-Algorithm/tree/main/Datasets/Real_Datasets) The real datasets are from the UCI repository of machine learning databases.
- The real data set includes Glass, Heartstatlog, lonosphere, Iris and heart.


### Synthetic Datasets
- [Synthetic_Datasets](https://github.com/Chw000/A-Hybrid-Clustering-Method-Based-On-K-means-Algorithm/tree/main/Datasets/Synthetic_Datasets) The synthetic datasets are from the CSDN datasets.


### How to use Datasets
- 1.Download the dataset you need
- 2.Change the bold text in the image below to the address you downloaded from
```
X = txt2array("**D:/python program/txt/Ionosphere.txt**", ",")
```


## Code
You can find [HTK-Clustering](https://github.com/Chw000/A-Hybrid-Clustering-Method-Based-On-K-means-Algorithm/blob/main/Code/HTK_Clustering.py) algorithm code and its baseline([K-means Clustering](https://github.com/Chw000/A-Hybrid-Clustering-Method-Based-On-K-means-Algorithm/blob/main/Code/K-means.py) and [Hierarchical Clustering](https://github.com/Chw000/A-Hybrid-Clustering-Method-Based-On-K-means-Algorithm/blob/main/Code/Hierarchical.py))


## Requirements

To install requirements:

```setup
pip install numpy
pip install psutil
pip install scikit-learn
pip install scipy
```
- Experimental operating system is Window XP,program language is Python 3.9.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

