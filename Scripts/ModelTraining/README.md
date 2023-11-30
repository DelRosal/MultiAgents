# Model Training Installation Guide

The Model Training module is intended to download a specific dataset, in this case a dataset with images of trucks classified as either empty or full. This module utilizes the YOLOv8 model for training and provides basic metrics to assess its performance.

## Prerequisites

Ensure that Python is installed on your system.

## Setting up a Virtual Environment

1. Create a virtual environment named "venv."

```sh
python -m venv venv
```

2. Activate the virtual environment.

```sh
source venv/bin/activate
```

## Install Dependencies

Install the required dependencies by executing the following command:

```sh
pip install -r requirements.txt
```

## Fine-Tuning

Review the ***fine-tuning.ipynb*** file, which is essential for downloading the dataset and obtaining training results. Results will be saved in the ./runs/detect folder.

## Prediction

After completing the training, run the ***prediction.ipynb*** file to assess its performance.

## Validation

Evaluate the prediction results to gauge the model's performance. Explore the validation metrics and ensure the model meets the desired standards.