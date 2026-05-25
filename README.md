# Sleep Staging from Instantaneous Heart Rate and Accelerometry

## Project Overview
This repository contains the R&D codebase for building an advanced deep learning model to perform robust sleep staging using non-invasive wearable data (Apple Watch Instantaneous Heart Rate and multi-axis Accelerometry).

The core objective is to exceed the performance of prior baseline papers on this dataset by addressing spatial-temporal gaps and employing hybrid model architectures (e.g., CNN + BiGRU + Attention).

## Dataset
The project operates on "A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels." The raw data includes physiological readings (HR, Acc) synchronized against Gold-Standard PolySomnoGraphy (PSG) derived EEG labels.

## Repository Structure
- `dataset/` - Directory for unzipped raw CSV/EDF data files.
- `pipeline.ipynb` - End-to-end jupyter notebook handling data ingestion, preprocessing, modeling, training, and evaluation.
- `context.md` - Research notes, decision logs, and references to problem statements and gaps.

## Usage
1. Ensure the dataset is unzipped into `dataset/` (Extract the 1.0.0 folder out of the zip so that the structure mimics `dataset/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/BidslabXX`).
2. Install necessary Python dependencies (`pip install torch torchvision pandas scipy scikit-learn seaborn matplotlib`).
3. Open `pipeline.ipynb` in VS Code / Jupyter Lab.
4. Run the cells sequentially to build the Data Loaders, train the Deep Learning mapping, and output the evaluations (Acc/Kappa Score Matrix).
