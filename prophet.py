from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

class ProphetMaster:
    def __init__(self):
        self.model = None
        self.forecast = None
    
    def prepare_data(self, df):
        """Prepare data in Prophet format"""
        prophet_df = df.reset_index()
        prophet_df = prophet_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        return prophet_df
    
    def create_super_prophet(self, df):
        """Create a Prophet model that will shock everyone with accuracy"""
        # Initialize with advanced parameters
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,  # Detect trend changes
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            interval_width=0.95
        )
        
        # Add custom seasonalities for stock market
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        # Add market regime indicators as regressors
        df['market_stress'] = df['volatility']
        df['volume_surge'] = (df['Volume'] > df['Volume'].rolling(30).mean()).astype(int)
        
        model.add_regressor('market_stress')
        model.add_regressor('volume_surge')
        
        return model
    
    def fit_and_predict(self, df, periods=60):
        """Fit Prophet and generate mind-blowing forecasts"""
        prophet_data = self.prepare_data(df)
        
        # Add regressors to prophet data
        prophet_data['market_stress'] = df['volatility'].values
        prophet_data['volume_surge'] = (df['Volume'] > df['Volume'].rolling(30).mean()).astype(int).values
        
        self.model = self.create_super_prophet(prophet_data)
        self.model.fit(prophet_data)
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        
        # Extend regressors for future predictions
        future['market_stress'] = prophet_data['market_stress'].iloc[-1]  # Use last known value
        future['volume_surge'] = 0  # Assume normal volume
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        return forecast, self.model