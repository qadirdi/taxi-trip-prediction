import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class TaxiDataProcessor:
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.valid_locations = set()
        self.location_probabilities = {}
        self.location_distances = defaultdict(dict)  # Store average distances between locations
        self.time_slot_minutes = 30  # Group times into 30-minute slots

    def load_and_process_data(self, data_file="data/Cleaned_Green_Taxi_Data.csv"):
        """Load and process the taxi trip data with enhanced location handling."""
        try:
            # Load data
            self.logger.info(f"Loading data from {data_file}")
            df = pd.read_csv(data_file)
            initial_size = len(df)
            self.logger.info(f"Initial number of records: {initial_size}")
            
            # Store valid locations
            self.valid_locations = set(df['PULocationID'].unique()) | set(df['DOLocationID'].unique())
            self.logger.info(f"Number of valid locations: {len(self.valid_locations)}")
            
            # Calculate average distances between locations
            location_pairs = df.groupby(['PULocationID', 'DOLocationID'])['trip_distance'].mean()
            for (pu_loc, do_loc), avg_distance in location_pairs.items():
                self.location_distances[pu_loc][do_loc] = avg_distance
            
            # Create time slots
            df['time_slot'] = df.apply(
                lambda x: self.get_time_slot(x['pickup_hour'], x['pickup_minute']),
                axis=1
            )
            
            # Calculate location probabilities for each pickup location and time slot
            self.location_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            
            for _, group in df.groupby(['PULocationID', 'time_slot']):
                pu_loc = group['PULocationID'].iloc[0]
                time_slot = group['time_slot'].iloc[0]
                total_trips = len(group)
                
                dropoff_counts = group['DOLocationID'].value_counts()
                for do_loc, count in dropoff_counts.items():
                    self.location_probabilities[pu_loc][time_slot][do_loc] = count / total_trips
            
            # Clean and filter data
            df = df[
                (df['trip_distance'] > 0) &
                (df['trip_distance'] < 50) &     # Reasonable distance range
                (df['total_amount'] > 0) &
                (df['total_amount'] < 200)       # Reasonable fare range
            ]
            
            # Prepare features
            features = np.column_stack([
                df['PULocationID'],
                df['pickup_hour'],
                df['pickup_day'],
                df['pickup_month'],
                df['trip_distance'],
                df['fare_amount'],
                df['tip_amount']
            ])
            
            # Scale features (excluding location IDs)
            scaled_features = features.copy()
            scaled_features[:, 1:] = self.feature_scaler.fit_transform(features[:, 1:])
            
            # Prepare targets
            targets = np.column_stack([
                df['DOLocationID'],
                df['dropoff_hour'],
                df['total_amount']
            ])
            
            # Scale targets (excluding location IDs)
            scaled_targets = targets.copy()
            scaled_targets[:, 1:] = self.target_scaler.fit_transform(targets[:, 1:])
            
            self.logger.info(f"Number of records after cleaning: {len(df)}")
            
            return scaled_features, {
                'dropoff_location': scaled_targets[:, 0],
                'dropoff_time': scaled_targets[:, 1],
                'earnings': scaled_targets[:, 2]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def get_time_slot(self, hour, minute):
        """Convert hour and minute to a time slot index (48 slots per day)."""
        total_minutes = hour * 60 + minute
        return total_minutes // self.time_slot_minutes  # This will give 48 slots (0-47)

    def prepare_prediction_input(self, features):
        """Prepare input features for prediction."""
        try:
            # Ensure features is a 2D array
            features = np.array(features).reshape(1, -1)
            
            # Create a copy to avoid modifying the original
            scaled_features = features.copy()
            
            # Scale all features except the location ID (first column)
            scaled_features[:, 1:] = self.feature_scaler.transform(features[:, 1:])
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction input: {str(e)}")
            raise
            
    def inverse_transform_predictions(self, predictions, exclude_locations=True):
        """Convert scaled predictions back to original scale."""
        try:
            if exclude_locations:
                # Don't inverse transform location predictions
                return np.column_stack([
                    predictions[:, 0],  # Keep locations as is
                    self.target_scaler.inverse_transform(predictions[:, 1:])
                ])
            return self.target_scaler.inverse_transform(predictions)
        except Exception as e:
            self.logger.error(f"Error inverse transforming predictions: {str(e)}")
            raise

    def get_likely_dropoff_locations(self, pickup_loc, hour, minute, top_k=5):
        """Get the most likely dropoff locations for a given pickup location and time."""
        if pickup_loc not in self.valid_locations:
            return []
            
        time_slot = self.get_time_slot(hour, minute)
        
        # Get probabilities for this pickup location and time slot
        dropoff_probs = self.location_probabilities[pickup_loc][time_slot]
        if not dropoff_probs:
            return []
            
        # Sort locations by probability
        sorted_locations = sorted(
            dropoff_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top K locations
        return [loc for loc, _ in sorted_locations[:top_k]]