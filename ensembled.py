class UltimateEnsemble:
    def __init__(self, df):
        self.df = df
        self.arima_model = AdvancedARIMA(df)
        self.prophet_model = ProphetMaster()
        self.lstm_model = LSTMPredictor()
        self.results = {}
    
    def run_all_models(self, forecast_days=30):
        """Run ALL models and create the ultimate ensemble prediction"""
        print("🔥 LAUNCHING THE ULTIMATE FORECASTING ARSENAL 🔥")
        
        # 1. ARIMA/SARIMA
        print("\n📈 Running Advanced ARIMA...")
        arima_forecast, arima_conf, arima_fitted = self.arima_model.fit_and_forecast(
            self.df['Close'], steps=forecast_days
        )
        
        # 2. Prophet
        print("\n🔮 Unleashing Prophet...")
        prophet_forecast, prophet_model = self.prophet_model.fit_and_predict(
            self.df, periods=forecast_days
        )
        
        # 3. LSTM
        print("\n🤖 Training LSTM Neural Network...")
        lstm_train, lstm_test, y_train, y_test, lstm_history = self.lstm_model.train_predict(self.df)
        
        # Store results
        self.results = {
            'arima': {'forecast': arima_forecast, 'confidence': arima_conf, 'model': arima_fitted},
            'prophet': {'forecast': prophet_forecast, 'model': prophet_model},
            'lstm': {'train_pred': lstm_train, 'test_pred': lstm_test, 'history': lstm_history}
        }
        
        return self.results
    
    def create_ensemble_forecast(self, forecast_days=30):
        """Combine all models into the ultimate prediction"""
        # Get individual forecasts
        arima_forecast = self.results['arima']['forecast']
        prophet_forecast = self.results['prophet']['forecast']['yhat'].tail(forecast_days).values
        
        # For LSTM, we need to predict future values
        last_sequence = self.df['Close'].tail(self.lstm_model.sequence_length).values
        lstm_future = self.predict_lstm_future(last_sequence, forecast_days)
        
        # Create weighted ensemble (you can adjust weights based on performance)
        ensemble_weights = {'arima': 0.3, 'prophet': 0.4, 'lstm': 0.3}
        
        ensemble_forecast = (
            ensemble_weights['arima'] * arima_forecast +
            ensemble_weights['prophet'] * prophet_forecast +
            ensemble_weights['lstm'] * lstm_future
        )
        
        return ensemble_forecast
    
    def predict_lstm_future(self, last_sequence, forecast_days):
        """Generate LSTM predictions for future days"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            # Scale current sequence
            scaled_sequence = self.lstm_model.scaler.transform(current_sequence.reshape(-1, 1))
            
            # Reshape for prediction
            input_seq = scaled_sequence[-self.lstm_model.sequence_length:].reshape(1, -1, 1)
            
            # Predict next value
            next_pred_scaled = self.lstm_model.model.predict(input_seq, verbose=0)
            next_pred = self.lstm_model.scaler.inverse_transform(next_pred_scaled)[0, 0]
            
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        return np.array(predictions)