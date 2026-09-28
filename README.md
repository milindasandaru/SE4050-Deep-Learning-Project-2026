# SE4050 Deep Learning: Phishing Website Detection

## Overview
This repository contains an end-to-end deep learning framework designed to detect phishing websites using engineered URL and webpage features. We benchmark four distinct deep learning architectures under identical experimental conditions to critically evaluate their predictive performance, generalization, and computational efficiency for the SE4050 assignment. 

The four architectures implemented are:
1. Multi-Layer Perceptron (MLP)
2. 1D Convolutional Neural Network (1D CNN)
3. Long Short-Term Memory (LSTM)
4. Gated Recurrent Unit (GRU)

## Dataset
**Dataset:** [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI Machine Learning Repository)
*   The project strictly utilizes 50 engineered numerical features. 
*   High-cardinality text attributes (e.g., `FILENAME`, `URL`, `Domain`) were excluded to maintain a controlled tabular environment.
*   To prevent data leakage, all preprocessing transformations (e.g., `StandardScaler`) were fitted exclusively on the training split before transforming the validation and unseen test sets.

## Repository Structure
*   `data/`: Contains instructions for dataset acquisition and the processed `.npy` arrays. *(Note: The raw CSV is ignored via `.gitignore` to prevent repository bloat).*
*   `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (`01_EDA.ipynb`), Preprocessing (`02_preprocessing.ipynb`), and individual model training.
*   `src/`: Shared Python scripts for modular scaling pipelines and unified evaluation metrics.
*   `models/`: Serialized model weights (`.h5`) and architecture definitions.
*   `results/`: Exported confusion matrices, ROC curves, loss/accuracy learning curves, and evaluation CSVs.
*   `requirements.txt`: Python package dependencies required for exact reproducibility[cite: 2].
*   `report/`: Final PDF documentation and Turnitin similarity reports.

## Setup and Execution Instructions
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/se4050-phishing-deep-learning.git](https://github.com/your-username/se4050-phishing-deep-learning.git)
   cd se4050-phishing-deep-learning