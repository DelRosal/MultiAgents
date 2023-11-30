# Truck Tracker Installation Guide

The Truck Tracker module is intended to process camera images to detect trucks, determine their direction, and classify their load status. Follow these steps to install and run the project:

## Prerequisites

Ensure you have Python installed on your system.

## Setting up a Virtual Environment

1. Create a virtual environment named "venv"

```sh
python -m venv venv
```

2. Activate the virtual environment.

```sh
source venv/bin/activate
```

## Install Dependencies

Install the required dependencies by running the following command:

```sh
pip install -r requirements.txt
```

## Run the Truck Tracker

Once the virtual environment is activated and dependencies are installed, execute the project by typing:

```sh
python3 -m main
```

This will initiate the Truck Tracker module, processing camera images and providing insights on detected trucks, their direction, and load status.