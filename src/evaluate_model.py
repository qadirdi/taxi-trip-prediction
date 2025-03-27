import numpy as np
import tensorflow as tf
from taxi_prediction_model import TaxiTripPredictor
import logging
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load historical data
historical_data = pd.read_csv('data/Cleaned_Green_Taxi_Data.csv')

def get_historical_dropoff_locations(current_loc, current_time, predictor):
    """Get likely dropoff locations based on historical patterns and model predictions"""
    # Get the data processor from the predictor
    data_processor = predictor.data_processor
    
    # Get likely locations based on historical probabilities
    likely_locations = data_processor.get_likely_dropoff_locations(
        current_loc,
        current_time.hour,
        current_time.minute
    )
    
    if not likely_locations:
        # Fallback to nearby locations if no historical data
        nearby_locs = [
            loc for loc in data_processor.valid_locations
            if abs(loc - current_loc) <= 10 and loc != current_loc
        ]
        return sorted(nearby_locs)[:5]
    
    return likely_locations

def calculate_distance(loc1, loc2, predictor=None):
    """Calculate approximate distance between two location IDs"""
    if predictor is None:
        # Fallback to simple calculation if predictor not provided
        return abs(loc1 - loc2) * 0.1 + 2.0
        
    # Get historical average distance if available
    historical_distance = predictor.data_processor.location_distances.get(loc1, {}).get(loc2)
    if historical_distance is not None:
        return historical_distance
        
    # Fallback to simple calculation
    return abs(loc1 - loc2) * 0.1 + 2.0

def predict_next_trip(current_loc, current_time, predictor):
    """Predict the most likely next trip using the trained model and historical patterns"""
    potential_locations = get_historical_dropoff_locations(current_loc, current_time, predictor)
    
    best_score = -float('inf')
    best_prediction = None
    best_features = None
    
    # Keep track of recent locations to avoid loops
    recent_locations = getattr(predictor, 'recent_locations', [])
    if not hasattr(predictor, 'recent_locations'):
        predictor.recent_locations = recent_locations
    
    for next_loc in potential_locations:
        distance = calculate_distance(current_loc, next_loc, predictor)
        base_fare = 2.50 + (distance * 2.50)
        expected_tip = base_fare * 0.15
        
        features = np.array([[
            current_loc,
            current_time.hour,
            current_time.day,
            current_time.month,
            distance,
            base_fare,
            expected_tip
        ]])
        
        predictions = predictor.predict(features)
        earnings = predictions['earnings'][0]
        duration = estimate_trip_duration(distance, current_time.hour)
        
        # Get probability of this dropoff location
        time_slot = predictor.data_processor.get_time_slot(current_time.hour, current_time.minute)
        location_probability = predictor.data_processor.location_probabilities[current_loc][time_slot].get(next_loc, 0)
        
        # Calculate diversity penalty if location was recently visited
        diversity_penalty = 1.0
        if next_loc in recent_locations:
            diversity_penalty = 0.5  # Reduce score by 50% if location was recently visited
        
        # Calculate rush hour bonus
        rush_hour_bonus = 1.0
        if (7 <= current_time.hour <= 10) or (16 <= current_time.hour <= 19):
            rush_hour_bonus = 1.2  # 20% bonus during rush hours
        
        # Calculate distance bonus (prefer medium-distance trips)
        distance_bonus = 1.0
        if 3 <= distance <= 8:  # Medium distance trips
            distance_bonus = 1.2
        elif distance > 8:  # Long distance trips
            distance_bonus = 1.1
        
        # Combined score considering all factors
        score = (
            (earnings / duration if duration > 0 else 0) *  # Base earnings per hour
            (1 + location_probability) *                    # Historical probability
            diversity_penalty *                            # Penalty for recent locations
            rush_hour_bonus *                             # Rush hour bonus
            distance_bonus                                # Distance preference
        )
        
        if score > best_score:
            best_score = score
            best_prediction = predictions
            best_features = {
                'dropoff_loc': next_loc,
                'distance': distance,
                'base_fare': base_fare,
                'expected_tip': expected_tip,
                'duration': duration,
                'probability': location_probability
            }
    
    # Update recent locations
    if best_features:
        recent_locations.append(best_features['dropoff_loc'])
        if len(recent_locations) > 5:  # Keep track of last 5 locations
            recent_locations.pop(0)
        predictor.recent_locations = recent_locations
    
    return best_prediction, best_features

def estimate_trip_duration(distance, hour, weather_factor=1.0):
    """Estimate trip duration in hours based on multiple factors"""
    # Base speed factors
    base_speed = 25.0  # base speed in mph
    
    # Time of day factors
    if 7 <= hour <= 10:  # Morning rush
        time_factor = 0.7  # 30% slower
    elif 16 <= hour <= 19:  # Evening rush
        time_factor = 0.6  # 40% slower
    elif 23 <= hour or hour <= 4:  # Late night
        time_factor = 1.3  # 30% faster
    else:
        time_factor = 1.0
    
    # Calculate effective speed
    effective_speed = base_speed * time_factor * weather_factor
    
    # Add traffic lights and stops factor
    base_duration = distance / effective_speed
    stops_factor = 1.2  # 20% additional time for stops
    
    return base_duration * stops_factor

def simulate_shift():
    """Simulate an 8-hour shift starting with a manual first order"""
    try:
        # Load and initialize the model
        predictor = TaxiTripPredictor()
        predictor.load_model()
        
        # Get manual input for first order
        logger.info("\nEnter details for the first order:")
        pickup_loc = int(input("Enter pickup location ID: "))
        dropoff_loc = int(input("Enter dropoff location ID: "))
        hour = int(input("Enter pickup hour (0-23): "))
        
        # Validate locations
        if pickup_loc not in predictor.data_processor.valid_locations:
            raise ValueError(f"Invalid pickup location ID: {pickup_loc}")
        if dropoff_loc not in predictor.data_processor.valid_locations:
            raise ValueError(f"Invalid dropoff location ID: {dropoff_loc}")
        
        # Initialize shift
        current_time = datetime.now().replace(hour=hour, minute=0)
        shift_end = current_time + timedelta(hours=8)
        total_earnings = 0
        trip_count = 0
        
        logger.info("\n" + "="*70)
        logger.info("8-HOUR SHIFT SIMULATION AND EARNINGS PREDICTION")
        logger.info("="*70)
        logger.info(f"Shift starts at: {current_time.strftime('%H:%M')}")
        logger.info(f"Shift ends at: {shift_end.strftime('%H:%M')}")
        
        # Process first order
        distance = calculate_distance(pickup_loc, dropoff_loc, predictor)
        base_fare = 2.50 + (distance * 2.50)
        expected_tip = base_fare * 0.15
        
        features = np.array([[
            pickup_loc,
            hour,
            current_time.day,
            current_time.month,
            distance,
            base_fare,
            expected_tip
        ]])
        
        predictions = predictor.predict(features)
        duration = estimate_trip_duration(distance, hour)
        current_loc = dropoff_loc
        
        # Log first trip
        logger.info(f"\nTrip {trip_count + 1}: Manual Input")
        logger.info("-"*50)
        logger.info("Trip Details:")
        logger.info(f"  Pickup Location: {pickup_loc}")
        logger.info(f"  Dropoff Location: {dropoff_loc}")
        logger.info(f"  Start Time: {current_time.strftime('%H:%M')}")
        logger.info(f"  Distance: {distance:.1f} miles")
        logger.info(f"  Base Fare: ${base_fare:.2f}")
        logger.info(f"  Predicted Earnings: ${predictions['earnings'][0]:.2f}")
        
        # Update time and earnings
        current_time += timedelta(hours=duration)
        current_time += timedelta(minutes=15)  # Buffer between trips
        total_earnings += predictions['earnings'][0]
        trip_count += 1
        
        # Predict subsequent trips until shift ends
        while current_time < shift_end:
            predictions, best_features = predict_next_trip(current_loc, current_time, predictor)
            
            if not best_features:
                logger.info("\nNo more viable trips found.")
                break
                
            trip_duration = best_features['duration']
            if current_time + timedelta(hours=trip_duration) > shift_end:
                logger.info("\nShift time limit reached.")
                break
                
            logger.info(f"\nTrip {trip_count + 1}: Predicted Optimal Route")
            logger.info("-"*50)
            logger.info("Trip Details:")
            logger.info(f"  Pickup Location: {current_loc}")
            logger.info(f"  Dropoff Location: {best_features['dropoff_loc']}")
            logger.info(f"  Start Time: {current_time.strftime('%H:%M')}")
            logger.info(f"  Distance: {best_features['distance']:.1f} miles")
            logger.info(f"  Base Fare: ${best_features['base_fare']:.2f}")
            logger.info(f"  Predicted Earnings: ${predictions['earnings'][0]:.2f}")
            logger.info(f"  Location Probability: {best_features['probability']:.3f}")
            
            # Update state
            current_time += timedelta(hours=trip_duration)
            current_time += timedelta(minutes=15)  # Buffer between trips
            total_earnings += predictions['earnings'][0]
            current_loc = best_features['dropoff_loc']
            trip_count += 1
        
        # Summary
        total_time = (current_time - datetime.now().replace(hour=hour, minute=0)).total_seconds() / 3600
        logger.info("\n" + "="*50)
        logger.info("SHIFT SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Trips: {trip_count}")
        logger.info(f"Total Earnings: ${total_earnings:.2f}")
        logger.info(f"Total Time: {total_time:.1f} hours")
        logger.info(f"Average Per Hour: ${total_earnings/total_time:.2f}")
        logger.info(f"Average Per Trip: ${total_earnings/trip_count:.2f}")
        logger.info("="*50)
            
    except Exception as e:
        logger.error(f"Error during shift simulation: {str(e)}")
        raise

if __name__ == "__main__":
    simulate_shift()