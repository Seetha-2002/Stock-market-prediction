class Config:
        SEQUENCE_LENGTH = 60  # Number of time steps to look back
        BATCH_SIZE = 32
        EPOCHS = 50
        LEARNING_RATE = 0.001
        TRAIN_SPLIT = 0.8
        FEATURE_COLUMNS = ['close', 'volume', 'open', 'high', 'low']
        TARGET_COLUMN = 'close'
        MODEL_PATH = 'saved_models/lstm_model.h5'
        DATA_PATH = '/content/drive/MyDrive/archive (10)/infolimpioavanzadoTarget.csv'
