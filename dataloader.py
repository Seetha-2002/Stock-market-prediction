import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

class DataLoader:
    """Data loader for stock market prediction."""
    
    def __init__(self, filename: str):
        """Initialize data loader.
        
        Args:
            filename (str): Path to the CSV file containing stock data
        """
        self.filename = filename
        self.scaler = MinMaxScaler()
        self.data = None
        self.scaled_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load stock data from CSV file."""
        self.data = pd.read_csv(self.filename)
        return self.data
    
    def prepare_data(self, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model.
        
        Args:
            sequence_length (int): Number of time steps to look back
            
        Returns:
            Tuple containing X (features) and y (targets) arrays
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
        self.scaled_data = scaled_data
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Convert scaled values back to original scale."""
        return self.scaler.inverse_transform(data)
