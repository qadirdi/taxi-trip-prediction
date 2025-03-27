# Taxi Trip Prediction System

This project implements a machine learning system for predicting taxi trip details including dropoff locations, trip duration, and earnings. The system uses historical taxi trip data to make intelligent predictions and help optimize taxi driver earnings.

## Features

- Predicts optimal dropoff locations based on pickup location and time
- Estimates trip duration considering traffic patterns and time of day
- Calculates expected earnings for each trip
- Simulates full 8-hour shifts with route optimization
- Uses historical data for accurate distance calculations

## Project Structure

```
.
├── src/
│   ├── data_processor.py      # Data processing and feature engineering
│   ├── taxi_prediction_model.py # Neural network model definition
│   └── evaluate_model.py      # Evaluation and simulation code
├── data/
│   └── Cleaned_Green_Taxi_Data.csv  # Historical taxi trip data
├── models/
│   └── best_model.keras      # Trained model weights
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required data file in the `data/` directory.

3. Make sure the trained model is in the `models/` directory.

## Usage

To run a shift simulation:

```bash
python src/evaluate_model.py
```

The simulation will:
1. Ask for initial trip details (pickup location, dropoff location, start hour)
2. Simulate an 8-hour shift
3. Provide detailed predictions for each trip
4. Show a summary of total earnings and statistics

## Model Details

The system uses a deep neural network with:
- Input features: pickup location, time, trip distance, base fare, tip
- Output predictions: dropoff location, dropoff time, earnings
- Multiple dense layers with batch normalization and dropout
- Trained on historical taxi trip data

## Dependencies

- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn 