import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, List

class StockPredictor:
    """LSTM model for stock price prediction."""
    
    def __init__(self, sequence_length: int):
        """Initialize the LSTM model.
        
        Args:
            sequence_length (int): Number of time steps to look back
        """
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """Build and compile LSTM model."""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             validation_split: float = 0.2, epochs: int = 100, 
             batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'StockPredictor':
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(filepath)
        instance = cls(model.input_shape[1])
        instance.model = model
        return instance
