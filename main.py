from data.data_loader import DataLoader
from model.lstm_model import StockPredictor
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Initialize data loader
    data_loader = DataLoader('data/stock_data.csv')
    data = data_loader.load_data()
    
    # Prepare data
    sequence_length = 60
    X, y = data_loader.prepare_data(sequence_length)
    
    # Reshape data for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Initialize and train model
    model = StockPredictor(sequence_length)
    history = model.train(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Convert predictions back to original scale
    predictions = data_loader.inverse_transform(predictions)
    actual_values = data_loader.inverse_transform(y_test.reshape(-1, 1))
    
    # Save model
    model.save_model('models/stock_predictor.h5')

if __name__ == "__main__":
    main()
