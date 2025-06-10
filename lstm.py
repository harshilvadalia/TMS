import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class LSTMPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
    
    def prepare_lstm_data(self, data, target_col='Close'):
        """Prepare data for LSTM - this is where magic happens"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_col]])
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_advanced_lstm(self, input_shape):
        """Build an LSTM that will absolutely shock everyone"""
        model = Sequential([
            # First LSTM layer with return sequences
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Dropout(0.2),
            
            # Second LSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for final prediction
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_predict(self, data, test_size=0.2, epochs=100):
        """Train LSTM and generate predictions that will blow minds"""
        X, y = self.prepare_lstm_data(data)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model
        self.model = self.build_advanced_lstm((X_train.shape[1], 1))
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train the model
        print("🚀 Training the LSTM beast...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Generate predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        train_pred = self.scaler.inverse_transform(train_pred)
        test_pred = self.scaler.inverse_transform(test_pred)
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        return train_pred, test_pred, y_train_actual, y_test_actual, history