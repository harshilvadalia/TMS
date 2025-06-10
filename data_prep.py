import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Advanced feature engineering
def create_advanced_features(df):
    """Create mind-blowing features that will shock everyone"""
    df = df.copy()
    
    # Technical indicators that pros use
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = df['EMA_12'] - df['Close'].ewm(span=26).mean()
    
    # Volatility features
    df['volatility'] = df['Daily Return'].rolling(window=30).std()
    df['price_momentum'] = df['Close'].pct_change(periods=5)
    
    # Lag features for pattern recognition
    for lag in [1, 2, 3, 7, 14]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    
    return df

def calculate_rsi(prices, window=14):
    """Calculate RSI - a powerful technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))