# Neural Collaborative Filtering (NCF)

## Overview
This project implements Neural Collaborative Filtering (NCF) on the MovieLens 1M dataset to predict user-item interactions. The model combines linear (GMF) and nonlinear (MLP) representations to capture complex interaction patterns.

## Features
- Implicit feedback conversion (ratings >= 4)
- Negative sampling
- Neural Collaborative Filtering (GMF + MLP)
- Training with early stopping
- Full-ranking evaluation
- Metrics: Recall@10 and NDCG@10

## Dataset
We use the MovieLens 1M dataset:
- Users: 6038
- Items: 3533
- Split:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

## Model
The model consists of:
- GMF branch (linear interactions)
- MLP branch (nonlinear interactions)
- Fusion layer combining both outputs

## Training Setup
- Loss: Binary Cross-Entropy (BCE)
- Optimizer: Adam
- Negative sampling ratio: 2 and 4
- Early stopping based on validation loss

## Evaluation
Full-ranking evaluation is used:
- Recall@10
- NDCG@10

## Results
Best model performance:
- Recall@10 ≈ 0.0845
- NDCG@10 ≈ 0.1289

## Project Structure
.
├── Recommender_System_1.ipynb
├── train.csv
├── val.csv
├── test.csv
├── dataset_info.json
├── best_ncf_model.pth
├── loss.png
└── README.md

## How to Run
1. Install dependencies:
pip install pandas numpy scikit-learn torch matplotlib

2. Run:
Recommender_System_1.ipynb

## Authors
- Adi Yusuf Arrasyid
- Roshun
- Emanual Ashong
