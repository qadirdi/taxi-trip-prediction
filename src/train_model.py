import os
import numpy as np
from data_processor import TaxiDataProcessor
from taxi_prediction_model import TaxiTripPredictor
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_taxi_predictor():
    """Train the taxi trip prediction model."""
    logging.info("Loading and processing data...")
    data_processor = TaxiDataProcessor()
    X, y_dict = data_processor.load_and_process_data()
    
    # Split into training and testing sets
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42
    )
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    y_train_dict = {
        'dropoff_location': y_dict['dropoff_location'][train_idx],
        'dropoff_time': y_dict['dropoff_time'][train_idx],
        'earnings': y_dict['earnings'][train_idx]
    }
    
    y_test_dict = {
        'dropoff_location': y_dict['dropoff_location'][test_idx],
        'dropoff_time': y_dict['dropoff_time'][test_idx],
        'earnings': y_dict['earnings'][test_idx]
    }
    
    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Test set size: {len(X_test)}")
    
    try:
        predictor = TaxiTripPredictor()
        predictor.build_model()
        predictor.compile_model()
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        history = predictor.train(
            X_train=X_train,
            y_train=y_train_dict,
            X_val=X_test,
            y_val=y_test_dict,
            epochs=50,
            batch_size=32
        )
        
        logging.info("Model training completed successfully")
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot location MAE
        plt.subplot(1, 3, 2)
        plt.plot(history.history['dropoff_location_mae'], label='Training MAE')
        plt.plot(history.history['val_dropoff_location_mae'], label='Validation MAE')
        plt.title('Dropoff Location MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Plot earnings MAE
        plt.subplot(1, 3, 3)
        plt.plot(history.history['earnings_mae'], label='Training MAE')
        plt.plot(history.history['val_earnings_mae'], label='Validation MAE')
        plt.title('Earnings MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_taxi_predictor() 