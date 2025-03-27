import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
import os
from data_processor import TaxiDataProcessor

logger = logging.getLogger(__name__)

class TaxiTripPredictor:
    def __init__(self):
        self.model = None
        self.data_processor = TaxiDataProcessor()
        
    def build_model(self):
        """Build the neural network model for trip prediction"""
        inputs = keras.layers.Input(shape=(7,))  # All features combined
        
        # Deep layers
        x = keras.layers.Dense(256, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output layers
        dropoff_location = keras.layers.Dense(1, name='dropoff_location')(x)
        dropoff_time = keras.layers.Dense(1, name='dropoff_time')(x)
        earnings = keras.layers.Dense(1, name='earnings')(x)
        
        self.model = keras.Model(
            inputs=inputs,
            outputs=[dropoff_location, dropoff_time, earnings]
        )

    def load_model(self, model_path='models/best_model.keras'):
        """Load a trained model and initialize data processor"""
        self.model = tf.keras.models.load_model(model_path)
        # Load and process data to initialize scalers
        self.data_processor.load_and_process_data()
        logger.info("Model and data processor initialized successfully")
        
    def predict(self, features):
        """Make predictions for the given features."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare features using data processor
        scaled_features = self.data_processor.prepare_prediction_input(features)
        predictions = self.model.predict(scaled_features)
        
        # Convert predictions back to original scale
        unscaled_predictions = self.data_processor.inverse_transform_predictions(
            np.column_stack([predictions[0], predictions[1], predictions[2]])
        )
        
        return {
            'dropoff_location': unscaled_predictions[:, 0],
            'dropoff_time': unscaled_predictions[:, 1],
            'earnings': unscaled_predictions[:, 2]
        } 