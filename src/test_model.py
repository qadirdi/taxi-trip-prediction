import numpy as np
from taxi_prediction_model import TaxiTripPredictor
from data_processor import TaxiDataProcessor
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    # Load the trained model
    predictor = TaxiTripPredictor()
    predictor.model = tf.keras.models.load_model('models/best_model.keras')
    
    # Create data processor
    data_processor = TaxiDataProcessor()
    
    # Sample test data (PULocationID, pickup_hour, pickup_day, pickup_month, trip_distance, fare_amount, tip_amount)
    test_features = np.array([
        [100, 14, 15, 1, 5.5, 20.0, 5.0],  # Sample trip 1
        [150, 8, 20, 1, 3.2, 15.0, 3.0],   # Sample trip 2
        [200, 18, 25, 1, 8.0, 30.0, 7.0]    # Sample trip 3
    ])
    
    # Scale the features
    scaled_features = data_processor.prepare_prediction_input(test_features)
    
    # Make predictions
    predictions = predictor.predict(scaled_features)
    
    # Convert predictions back to original scale
    predictions_array = np.column_stack([
        predictions['dropoff_location'],
        predictions['dropoff_time'],
        predictions['earnings']
    ])
    original_scale_predictions = data_processor.inverse_transform_predictions(predictions_array)
    
    # Print results
    for i in range(len(test_features)):
        logger.info(f"\nTest Trip {i+1}:")
        logger.info(f"Input Features:")
        logger.info(f"  Pickup Location: {test_features[i][0]}")
        logger.info(f"  Pickup Hour: {test_features[i][1]}")
        logger.info(f"  Trip Distance: {test_features[i][4]:.1f} miles")
        logger.info(f"\nPredictions:")
        logger.info(f"  Dropoff Location: {original_scale_predictions[i][0]:.0f}")
        logger.info(f"  Dropoff Hour: {original_scale_predictions[i][1]:.0f}")
        logger.info(f"  Total Earnings: ${original_scale_predictions[i][2]:.2f}")

if __name__ == "__main__":
    test_model() 