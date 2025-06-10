import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import itertools
import warnings
warnings.filterwarnings('ignore')

class AdvancedARIMA:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.forecast = None
        self.fitted_model = None
    
    def check_stationarity(self, series):
        """Check if series is stationary - crucial for ARIMA"""
        result = adfuller(series.dropna())
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        
        if result[1] <= 0.05:
            print("✅ Series is stationary!")
            return True
        else:
            print("❌ Series is NOT stationary - needs differencing")
            return False
    
    def make_stationary(self, series):
        """Transform series to make it stationary"""
        print("🔧 Making series stationary...")
        
        # First difference
        diff_series = series.diff().dropna()
        
        if self.check_stationarity(diff_series):
            return diff_series, 1
        
        # Second difference if needed
        diff2_series = diff_series.diff().dropna()
        if self.check_stationarity(diff2_series):
            return diff2_series, 2
            
        return diff2_series, 2
    
    def auto_find_best_params(self, series):
        """Automatically find the BEST parameters - OPTIMIZED VERSION!"""
        print("🔥 Auto-detecting optimal ARIMA parameters (FAST MODE)...")
        
        try:
            # OPTIMIZED: Use smaller seasonal periods and faster search
            model = auto_arima(series, 
                              start_p=0, start_q=0, 
                              max_p=3, max_q=3,  # Reduced from 5 to 3
                              seasonal=True, 
                              start_P=0, start_Q=0,
                              max_P=1, max_Q=1,  # Reduced from 2 to 1
                              m=12,  # Monthly seasonality instead of 252
                              max_d=2, max_D=1,
                              stepwise=True,
                              suppress_warnings=True,
                              error_action='ignore',
                              n_jobs=-1,  # Use all CPU cores
                              maxiter=50)  # Limit iterations
            
            print(f"🎯 Best model: ARIMA{model.order} x SARIMA{model.seasonal_order}")
            return model.order, model.seasonal_order
            
        except Exception as e:
            print(f"⚠️  Auto ARIMA failed: {e}")
            print("🔄 Falling back to manual parameter selection...")
            return self.manual_param_search(series)
    
    def manual_param_search(self, series):
        """Manual grid search for best parameters - FAST VERSION"""
        print("🔍 Performing FAST manual grid search...")
        
        # OPTIMIZED: Smaller parameter ranges
        p_values = range(0, 3)
        d_values = range(0, 3)
        q_values = range(0, 3)
        
        best_aic = float('inf')
        best_params = None
        
        print("Testing different ARIMA combinations...")
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                            print(f"  New best: ARIMA{best_params} (AIC: {best_aic:.2f})")
                            
                    except:
                        continue
        
        print(f"🎯 Best manual parameters: ARIMA{best_params} (AIC: {best_aic:.2f})")
        return best_params, (0, 0, 0, 0)  # No seasonal for manual search
    
    def fit_and_forecast(self, series, steps=30):
        """Fit model and generate shocking accurate forecasts"""
        print(f"\n📊 Fitting ARIMA model for {len(series)} data points...")
        
        # Auto-check and make stationary if needed
        if not self.check_stationarity(series):
            print("🔧 Automatically making series stationary...")
            stationary_series, d_order = self.make_stationary(series)
            print(f"✅ Series made stationary with {d_order} differences")
        else:
            stationary_series = series
        
        order, seasonal_order = self.auto_find_best_params(series)  # Use original series
        
        try:
            # Fit the model
            print("🚀 Training ARIMA model...")
            self.model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            self.fitted_model = self.model.fit()
            
            print(f"✅ Model trained successfully!")
            print(f"📈 AIC: {self.fitted_model.aic:.2f}")
            print(f"📉 BIC: {self.fitted_model.bic:.2f}")
            
            # Generate forecasts with confidence intervals
            print(f"🔮 Generating {steps}-day forecast...")
            forecast_result = self.fitted_model.forecast(steps=steps, alpha=0.05)
            confidence_intervals = self.fitted_model.get_forecast(steps=steps).conf_int()
            
            # Store forecast
            self.forecast = forecast_result
            
            return forecast_result, confidence_intervals, self.fitted_model
            
        except Exception as e:
            print(f"❌ Error fitting ARIMA model: {e}")
            return None, None, None
    
    def plot_diagnostics(self):
        """Plot model diagnostics - this will impress everyone!"""
        if self.fitted_model is None:
            print("❌ No fitted model available. Run fit_and_forecast first!")
            return
        
        print("📊 Generating diagnostic plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🔬 ARIMA Model Diagnostics', fontsize=16, fontweight='bold')
        
        # Residuals plot
        residuals = self.fitted_model.resid
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Check)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add normal curve to histogram
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        axes[1, 0].legend()
        
        # ACF of residuals
        from statsmodels.tsa.stattools import acf
        lags = min(40, len(residuals)//4)  # Adaptive lag selection
        autocorr = acf(residuals, nlags=lags)
        axes[1, 1].stem(range(lags+1), autocorr)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Autocorrelation of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print diagnostic summary
        print("\n📋 DIAGNOSTIC SUMMARY:")
        print(f"   Mean residual: {residuals.mean():.6f}")
        print(f"   Std residual: {residuals.std():.6f}")
        print(f"   Ljung-Box p-value: {self.fitted_model.test_serial_correlation('ljungbox')[0][1]:.4f}")
    
    def plot_forecast(self, original_series, forecast_steps=30):
        """Create stunning forecast visualization"""
        if self.forecast is None:
            print("❌ No forecast available. Run fit_and_forecast first!")
            return
        
        print("📈 Creating forecast visualization...")
        
        # Create future dates
        last_date = original_series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=forecast_steps, freq='D')
        
        # Get confidence intervals
        forecast_ci = self.fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        plt.figure(figsize=(16, 10))
        
        # Create subplots
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3)
        
        # Main forecast plot
        ax1 = plt.subplot(gs[0, :])
        
        # Plot historical data (last 200 points for better context)
        historical_data = original_series.tail(200)
        ax1.plot(historical_data.index, historical_data.values, 
                label='Historical Data', color='blue', linewidth=2, alpha=0.8)
        
        # Plot forecast
        ax1.plot(future_dates, self.forecast, 
                label='ARIMA Forecast', color='red', linewidth=3)
        
        # Plot confidence intervals
        ax1.fill_between(future_dates, 
                        forecast_ci.iloc[:, 0], 
                        forecast_ci.iloc[:, 1], 
                        color='red', alpha=0.2, label='95% Confidence Interval')
        
        ax1.set_title('🚀 ARIMA Stock Price Forecast', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Stock Price ($)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Forecast trend analysis
        ax2 = plt.subplot(gs[1, 0])
        daily_changes = np.diff(self.forecast)
        ax2.bar(range(len(daily_changes)), daily_changes, 
               color=['green' if x > 0 else 'red' for x in daily_changes], alpha=0.7)
        ax2.set_title('Daily Forecast Changes', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Days Ahead')
        ax2.set_ylabel('Price Change ($)')
        ax2.grid(True, alpha=0.3)
        
        # Forecast statistics
        ax3 = plt.subplot(gs[1, 1])
        stats_data = {
            'Mean': self.forecast.mean(),
            'Max': self.forecast.max(),
            'Min': self.forecast.min(),
            'Std': self.forecast.std()
        }
        bars = ax3.bar(stats_data.keys(), stats_data.values(), color=['blue', 'green', 'red', 'orange'])
        ax3.set_title('Forecast Statistics', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Price ($)')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_data.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*bar.get_height(), 
                    f'${value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print forecast summary
        print("\n" + "="*70)
        print("📊 ARIMA FORECAST SUMMARY")
        print("="*70)
        current_price = original_series.iloc[-1]
        next_day_pred = self.forecast.iloc[0]
        
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"🎯 Next Day Prediction: ${next_day_pred:.2f} ({((next_day_pred/current_price-1)*100):+.2f}%)")
        print(f"📈 7-Day Average: ${self.forecast[:7].mean():.2f}")
        print(f"📊 30-Day Trend: {'🚀 BULLISH' if self.forecast.iloc[-1] > current_price else '📉 BEARISH'}")
        print(f"📉 Forecast Volatility: ${self.forecast.std():.2f}")
        print(f"🎢 Price Range: ${self.forecast.min():.2f} - ${self.forecast.max():.2f}")
        print("="*70)
    
    def get_model_summary(self):
        """Get detailed model summary"""
        if self.fitted_model is None:
            print("❌ No fitted model available!")
            return None
        
        print("\n📋 ARIMA MODEL SUMMARY")
        print("="*50)
        print(self.fitted_model.summary())
        
        return self.fitted_model.summary()

# FAST test function
def test_arima():
    """Test the ARIMA implementation with REAL Apple data"""
    print("🧪 Testing ARIMA Implementation with REAL DATA...")
    
    try:
        # Load real Apple data
        df = pd.read_csv('cleaned_stock_market_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Use last 500 data points for faster testing
        df_test = df.tail(500)
        print(f"📊 Using {len(df_test)} data points from {df_test.index[0]} to {df_test.index[-1]}")
        
        # Test ARIMA
        arima_model = AdvancedARIMA(df_test)
        forecast, conf_int, fitted = arima_model.fit_and_forecast(df_test['Close'], steps=10)
        
        if forecast is not None:
            print("✅ ARIMA test passed!")
            arima_model.plot_forecast(df_test['Close'], forecast_steps=10)
            arima_model.plot_diagnostics()
        else:
            print("❌ ARIMA test failed!")
            
    except FileNotFoundError:
        print("📁 AAPL.csv not found, using synthetic data...")
        # Fallback to synthetic data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        
        # Generate realistic stock price data
        returns = np.random.normal(0.0005, 0.015, 500)  # More realistic parameters
        prices = [100]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        df = pd.DataFrame({'Date': dates, 'Close': prices[:-1]})
        df.set_index('Date', inplace=True)
        
        # Test ARIMA
        arima_model = AdvancedARIMA(df)
        forecast, conf_int, fitted = arima_model.fit_and_forecast(df['Close'], steps=10)
        
        if forecast is not None:
            print("✅ ARIMA test passed!")
            arima_model.plot_forecast(df['Close'], forecast_steps=10)
        else:
            print("❌ ARIMA test failed!")

if __name__ == "__main__":
    test_arima()