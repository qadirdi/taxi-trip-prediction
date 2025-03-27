import os
from datetime import datetime, timedelta
from taxi_prediction_model import TaxiTripPredictor
from data_processor import TaxiDataProcessor
import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripPredictor:
    def __init__(self, model_path='taxi_predictor'):
        """Initialize the trip predictor with a trained model"""
        self.model_path = model_path
        self.predictor = TaxiTripPredictor()
        self.data_processor = TaxiDataProcessor()
        self.load_model()
        
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            self.predictor.load_model(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def prepare_features(self, pickup_loc, dropoff_loc, pickup_time):
        """Prepare features for prediction"""
        # Create a sample dataframe
        df = pd.DataFrame({
            'PULocationID': [pickup_loc],
            'DOLocationID': [dropoff_loc],
            'lpep_pickup_datetime': [pd.to_datetime(pickup_time)],
            'lpep_dropoff_datetime': [pd.to_datetime(pickup_time) + pd.Timedelta(minutes=30)],  # Initial estimate
            'trip_distance': [5.0],  # Initial estimate
            'fare_amount': [20.0],  # Initial estimate
            'duration': [30.0]  # Initial estimate
        })
        
        # Process the sample data
        df = self.data_processor._clean_data(df)
        self.data_processor.calculate_location_stats(df)
        self.data_processor.calculate_temporal_stats(df)
        df = self.data_processor.calculate_trip_metrics(df)
        
        # Prepare feature groups
        location_features = np.column_stack([
            self.predictor.location_encoder.transform([pickup_loc]),
            [self.data_processor.location_stats[pickup_loc]['avg_distance']],
            [self.data_processor.location_stats[pickup_loc]['avg_fare']]
        ])
        
        temporal_features = np.column_stack([
            [df['lpep_pickup_datetime'].dt.hour.iloc[0]],
            [df['lpep_pickup_datetime'].dt.day.iloc[0]],
            [df['lpep_pickup_datetime'].dt.month.iloc[0]],
            [df['weekend'].iloc[0]],
            [df['rush_hour'].iloc[0]],
            [df['night'].iloc[0]],
            [df['time_window'].iloc[0]]
        ])
        
        trip_features = np.column_stack([
            [df['trip_distance'].iloc[0]],
            [df['duration'].iloc[0]],
            [df['fare_amount'].iloc[0]],
            [df['speed'].iloc[0]],
            [df['location_hour_fare'].iloc[0]]
        ])
        
        # Scale features
        location_features = self.predictor.location_scaler.transform(location_features)
        temporal_features = self.predictor.temporal_scaler.transform(temporal_features)
        trip_features = self.predictor.trip_scaler.transform(trip_features)
        
        return [location_features, temporal_features, trip_features]
        
    def predict_shift(self, pickup_loc, dropoff_loc, pickup_time):
        """Predict potential trips and earnings for an 8-hour shift"""
        logger.info(f"Predicting shift starting at {pickup_time} from location {pickup_loc}")
        
        shift_end = pd.to_datetime(pickup_time) + pd.Timedelta(hours=8)
        current_time = pd.to_datetime(pickup_time)
        current_loc = pickup_loc
        total_earnings = 0
        trips = []
        
        while current_time < shift_end:
            # Prepare features for current state
            features = self.prepare_features(current_loc, dropoff_loc, current_time)
            
            # Make prediction
            predictions = self.predictor.predict(features)
            next_loc = int(predictions[0][0])  # Predicted dropoff location
            trip_duration = int(predictions[1][0] * 60)  # Convert to minutes
            earnings = float(predictions[2][0])  # Earnings per minute
            
            # Calculate trip details
            trip_end_time = current_time + pd.Timedelta(minutes=trip_duration)
            trip_earnings = earnings * trip_duration
            
            # Add trip to sequence
            trips.append({
                'pickup_location': current_loc,
                'dropoff_location': next_loc,
                'start_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': trip_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': trip_duration,
                'earnings': trip_earnings
            })
            
            # Update state
            current_loc = next_loc
            current_time = trip_end_time + pd.Timedelta(minutes=5)  # 5-minute buffer between trips
            total_earnings += trip_earnings
            
            # Log progress
            logger.info(f"Predicted trip: {current_loc} -> {next_loc}, "
                       f"Duration: {trip_duration}min, Earnings: ${trip_earnings:.2f}")
            
        return {
            'total_earnings': total_earnings,
            'num_trips': len(trips),
            'trips': trips,
            'shift_start': pickup_time,
            'shift_end': shift_end.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def evaluate_order(self, pickup_loc, dropoff_loc, pickup_time):
        """Evaluate a potential order's impact on total earnings"""
        # Predict earnings without this order
        baseline = self.predict_shift(pickup_loc, pickup_loc, pickup_time)
        
        # Predict earnings with this order
        with_order = self.predict_shift(pickup_loc, dropoff_loc, pickup_time)
        
        # Calculate impact
        earnings_impact = with_order['total_earnings'] - baseline['total_earnings']
        trips_impact = with_order['num_trips'] - baseline['num_trips']
        
        return {
            'earnings_impact': earnings_impact,
            'trips_impact': trips_impact,
            'baseline_earnings': baseline['total_earnings'],
            'with_order_earnings': with_order['total_earnings'],
            'recommendation': 'Accept' if earnings_impact > 0 else 'Reject'
        }

def test_predictions():
    """Test the prediction system with sample cases"""
    predictor = TripPredictor()
    
    # Test morning shift
    logger.info("\nTesting morning shift predictions...")
    morning_result = predictor.predict_shift(
        pickup_loc=161,
        dropoff_loc=237,
        pickup_time='2023-01-04 07:00:00'
    )
    
    logger.info(f"Morning Shift Summary:")
    logger.info(f"Total Earnings: ${morning_result['total_earnings']:.2f}")
    logger.info(f"Number of Trips: {morning_result['num_trips']}")
    logger.info(f"Shift Period: {morning_result['shift_start']} to {morning_result['shift_end']}")
    
    # Test evening shift
    logger.info("\nTesting evening shift predictions...")
    evening_result = predictor.predict_shift(
        pickup_loc=142,
        dropoff_loc=236,
        pickup_time='2023-01-04 17:00:00'
    )
    
    logger.info(f"Evening Shift Summary:")
    logger.info(f"Total Earnings: ${evening_result['total_earnings']:.2f}")
    logger.info(f"Number of Trips: {evening_result['num_trips']}")
    logger.info(f"Shift Period: {evening_result['shift_start']} to {evening_result['shift_end']}")
    
    # Test order evaluation
    logger.info("\nTesting order evaluation...")
    eval_result = predictor.evaluate_order(
        pickup_loc=161,
        dropoff_loc=237,
        pickup_time='2023-01-04 08:00:00'
    )
    
    logger.info(f"Order Evaluation:")
    logger.info(f"Earnings Impact: ${eval_result['earnings_impact']:.2f}")
    logger.info(f"Trips Impact: {eval_result['trips_impact']}")
    logger.info(f"Recommendation: {eval_result['recommendation']}")

if __name__ == "__main__":
    test_predictions() 