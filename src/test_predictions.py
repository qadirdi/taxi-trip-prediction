import os
from datetime import datetime, timedelta
from predict_trips import TripPredictor
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_shift_predictions():
    """
    Test the model's predictions for a taxi driver's shift
    """
    try:
        # Initialize predictor
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "taxi_predictor")
        predictor = TripPredictor(model_path)
        
        # Test case 1: Morning shift starting from location 161
        current_order = {
            'pickup_location': 161,  # Example location ID
            'dropoff_location': 237,  # Example destination
            'pickup_time': '2023-01-04 07:00:00',  # Morning rush hour
            'fare': 25.0
        }
        
        logger.info("Testing morning shift predictions...")
        logger.info(f"Initial order: Starting from location {current_order['pickup_location']} at {current_order['pickup_time']}")
        
        predictions = predictor.predict_shift_earnings(current_order)
        
        # Print predictions in a readable format
        logger.info("\nPredicted shift summary:")
        logger.info(f"Total predicted earnings: ${predictions['total_predicted_earnings']:.2f}")
        logger.info(f"Number of predicted trips: {predictions['number_of_trips']}")
        logger.info(f"Shift end time: {predictions['shift_end_time']}")
        
        logger.info("\nPredicted trip sequence:")
        for trip in predictions['trip_sequence']:
            logger.info(
                f"Trip #{trip['order_number']}: "
                f"Location {trip['pickup']} → {trip['dropoff']}, "
                f"Time: {trip['time']}, "
                f"Expected earnings: ${trip['earnings']:.2f}"
            )
        
        # Test case 2: Evening shift starting from different location
        current_order = {
            'pickup_location': 142,  # Different starting location
            'dropoff_location': 236,
            'pickup_time': '2023-01-04 17:00:00',  # Evening rush hour
            'fare': 30.0
        }
        
        logger.info("\nTesting evening shift predictions...")
        logger.info(f"Initial order: Starting from location {current_order['pickup_location']} at {current_order['pickup_time']}")
        
        predictions = predictor.predict_shift_earnings(current_order)
        
        logger.info("\nPredicted shift summary:")
        logger.info(f"Total predicted earnings: ${predictions['total_predicted_earnings']:.2f}")
        logger.info(f"Number of predicted trips: {predictions['number_of_trips']}")
        logger.info(f"Shift end time: {predictions['shift_end_time']}")
        
        logger.info("\nPredicted trip sequence:")
        for trip in predictions['trip_sequence']:
            logger.info(
                f"Trip #{trip['order_number']}: "
                f"Location {trip['pickup']} → {trip['dropoff']}, "
                f"Time: {trip['time']}, "
                f"Expected earnings: ${trip['earnings']:.2f}"
            )
        
        # Test order evaluation
        logger.info("\nTesting order evaluation...")
        test_order = {
            'pickup_location': 161,
            'dropoff_location': 237,
            'pickup_time': '2023-01-04 08:30:00',
            'fare': 35.0
        }
        
        evaluation = predictor.evaluate_order(
            current_location=142,
            order_details=test_order,
            current_earnings=100.0,
            remaining_shift_time=timedelta(hours=6)
        )
        
        logger.info("\nOrder evaluation results:")
        logger.info(f"Order fare: ${evaluation['order_fare']:.2f}")
        logger.info(f"Predicted total with order: ${evaluation['predicted_total_with_order']:.2f}")
        logger.info(f"Predicted total without order: ${evaluation['predicted_total_without_order']:.2f}")
        logger.info(f"Earnings impact: ${evaluation['earnings_impact']:.2f}")
        logger.info(f"Recommendation: {evaluation['recommendation'].upper()}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_shift_predictions() 