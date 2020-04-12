# IASD NPM3D
IASD/MVA course on 3D cloud points

This repository contains the code, experiments results and report for the practical works (TP) and the project 
The goal of the project is to implement the paper "Pairwise Registration of TLS Point Clouds using Covariance Descriptors and a Non-cooperative Game" paper by Dawei Zai, Jonathan Li, Yulan Guo, Ming Cheng, Pengdi Huang, Xiaofei Cao, Cheng Wang
(https://www.sciencedirect.com/science/article/abs/pii/S0924271617303180). 

## Requirements

The project make use of several common Python libraries:
- `numpy`,`scipy`: scientific computing
- `matplotlib`,`seaborn`: results visualization 

## Data preparation

In order to reproduce the results:
- Download the dataset from the website Semantic3D (training set and labels for bildstein1 and bildstein3) http://www.semantic3d.net/view_dbase.php?chl=1
- Place the data in project/data directory and rename the files (bildstein%i)
- Launch sequentially in the file clean_data.py, the two subparts of the main with the adapted filename
- Launch, in features.py, the part on ACOV to get the ACOV descriptors. You may need to change the parameters if needed.
- Launch the code in matching.py to get the associations of keypoints kept by the non-cooperative game
- Launch transformation.py to obtain the translation and rotation, as well as the transformed cloudpoints

## Project structure and files

### Jupyter notebooks

- `computation analysis.ipynb`: Analysis of computation issues
- `keypoints.ipynb`:  analysis of keypoints

### Main algorithm

- `clean_data.py`: pre-processing for big datasets
- `features.py`: Implementation of ACOV
- `fileutils.py`: Utility file for managing filenames
- `matching.py`: Implementation of associations of keypoints (inclusive ratio, payoff matrix, non-cooperative games
- `subsampling.py`: Pre-processing for subsampling dataset
- `transformation.py`: Calculation of final transformation
  
### Data management

- `project/code/`: All the code and notebooks for the project
- `project/data/`: Dataset files location
- `project/images/`: Images saved and used for the report
- `project/saved_data/`: Intermediate and final processed data files location
