## An Interpretable Deep Learning and Molecular Docking Framework for Repurposing SARS-CoV-2 Main Protease Inhibitors

This repository provides a deep learning-based framework for predicting drug–target interactions (DTIs), built upon our previously proposed AMMVF model. 
It allows for training, evaluation, and visualization of DTI prediction.

## Requirements

Python ≥ 3.8
PyTorch ≥ 1.11
CUDA ≥ 11.3
RDKit ≥ 2020.09.1
NumPy ≥ 1.21.5
pandas ≥ 1.3.5

## Project Structure
├── train/
│   ├── train_featurizer.py
│   ├── model.py
│   └── train_main.py
├── attention/
│   ├── main_visualization.py
│   └── model_attention.py

## Usage
Part 1: Model Training (train/)

Run train_featurizer.py to extract features from the training set.
The model.py script contains the architecture and modules of the deep learning model.
Run train_main.py to train the model.

Part 2: Attention Visualization (attention/)

model_attention.py outputs attention weights.

main_visualization.py visualizes these weights.

Note: To visualize a specific test sample (e.g., Nirmatrelvir–6LU7 at index 1), update both model_attention.py and main_visualization.py to match the corresponding sample index (idx == 1 and data_list[1], respectively).

## Contact
For any questions or feedback, please contact the developer at chenqu@vip.126.com for further information.
