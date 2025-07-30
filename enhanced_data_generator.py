import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from datetime import datetime, timedelta
import warnings
import time
import random
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

class EnhancedCandlestickDataGenerator:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        self.chart_dir = os.path.join(output_dir, "enhanced_charts")
        self.label_file = os.path.join(output_dir, "enhanced_labels.csv")
        
        # Create directories
        os.makedirs(self.chart_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Expanded stock list with different sectors
        self.stocks = [
            # Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC",
            # Finance
            "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP",
            # Healthcare
            "JNJ", "PFE", "UNH", "MRK", "ABBV", "TMO", "DHR", "BMY",
            # Consumer
            "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "DIS",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG",
            # Industrial
            "BA", "GE", "CAT", "MMM", "HON"
        ]
        
    def calculate_technical_indicators(self, data):
        """Calculate enhanced technical indicators"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Enhanced features
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df['Price_Change_3'] = df['Close'].pct_change(3)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_7'] = df['Close'].pct_change(7)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Volume Price Trend
        df['VPT'] = (df['Volume'] * df['Close'].pct_change()).rolling(10).sum()
        
        # Bollinger Squeeze
        df['BB_Squeeze'] = df['BB_Width'] / df['BB_Middle']
        
        # RSI and MACD momentum
        df['RSI_Momentum'] = df['RSI'].diff(5)
        df['MACD_Momentum'] = df['MACD'].diff(3)
        
        # Support/Resistance levels
        df = self.find_support_resistance(df)
        
        # Market Regime Indicators
        df = self.add_market_regime_features(df)
        
        return df
    
    def find_support_resistance(self, df, window=5):
        """Enhanced support and resistance calculation"""
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find peaks and troughs
        peaks = argrelextrema(highs, np.greater, order=window)[0]
        troughs = argrelextrema(lows, np.less, order=window)[0]
        
        # Initialize columns
        df['Distance_to_Support'] = 0.0
        df['Distance_to_Resistance'] = 0.0
        df['Support_Strength'] = 0.0
        df['Resistance_Strength'] = 0.0
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            
            # Find nearest support (below current price)
            supports = df['Low'].iloc[troughs[troughs < i]]
            if len(supports) > 0:
                valid_supports = supports[supports <= current_price]
                if len(valid_supports) > 0:
                    nearest_support = valid_supports.max()
                    df.iloc[i, df.columns.get_loc('Distance_to_Support')] = (current_price - nearest_support) / current_price
                    # Support strength based on how many times it was tested
                    df.iloc[i, df.columns.get_loc('Support_Strength')] = len(valid_supports)
            
            # Find nearest resistance (above current price)
            resistances = df['High'].iloc[peaks[peaks < i]]
            if len(resistances) > 0:
                valid_resistances = resistances[resistances >= current_price]
                if len(valid_resistances) > 0:
                    nearest_resistance = valid_resistances.min()
                    df.iloc[i, df.columns.get_loc('Distance_to_Resistance')] = (nearest_resistance - current_price) / current_price
                    # Resistance strength
                    df.iloc[i, df.columns.get_loc('Resistance_Strength')] = len(valid_resistances)
        
        return df
    
    def add_market_regime_features(self, df):
        """Add market regime indicators"""
        # Volatility regime
        volatility_20_mean = df['Volatility_20'].rolling(20).mean()
        volatility_percentile = df['Volatility_20'].rolling(60).rank(pct=True)
        
        df['Volatility_Regime'] = np.where(
            volatility_percentile > 0.7, 2,  # High volatility
            np.where(volatility_percentile < 0.3, 0, 1)  # Low, Medium
        )
        
        # Trend regime
        df['Trend_Regime'] = np.where(
            (df['SMA_20'] > df['SMA_50']) & (df['Close'] > df['SMA_20']), 2,  # Strong bullish
            np.where(
                (df['SMA_20'] < df['SMA_50']) & (df['Close'] < df['SMA_20']), 0,  # Strong bearish
                1  # Sideways/Weak trend
            )
        )
        
        return df
    
    def enhanced_labeling(self, data, current_idx, future_days=5):
        """Enhanced labeling strategy with multiple factors"""
        if current_idx + future_days >= len(data):
            return None, {}
            
        current_data = data.iloc[current_idx]
        future_data = data.iloc[current_idx + future_days]
        
        current_price = current_data['Close']
        future_price = future_data['Close']
        
        # Basic price change
        price_change = (future_price - current_price) / current_price
        
        # Enhanced technical features
        features = {
            'price_change': price_change,
            'rsi': current_data.get('RSI', 50),
            'macd': current_data.get('MACD', 0),
            'bb_position': current_data.get('BB_Position', 0.5),
            'stoch_k': current_data.get('Stoch_K', 50),
            'volume_ratio': current_data.get('Volume_Ratio', 1),
            'volatility': current_data.get('Volatility', 0),
            'distance_to_support': current_data.get('Distance_to_Support', 0),
            'distance_to_resistance': current_data.get('Distance_to_Resistance', 0),
            'vpt': current_data.get('VPT', 0),
            'bb_squeeze': current_data.get('BB_Squeeze', 0),
            'rsi_momentum': current_data.get('RSI_Momentum', 0),
            'macd_momentum': current_data.get('MACD_Momentum', 0),
            'support_strength': current_data.get('Support_Strength', 0),
            'resistance_strength': current_data.get('Resistance_Strength', 0),
            'volatility_regime': current_data.get('Volatility_Regime', 1),
            'trend_regime': current_data.get('Trend_Regime', 1)
        }
        
        # Enhanced labeling logic
        label = self.determine_enhanced_label(price_change, features)
        
        return label, features
    
    def determine_enhanced_label(self, price_change, features):
        """Enhanced multi-factor labeling logic"""
        # Base thresholds
        buy_threshold = 0.02  # 2%
        sell_threshold = -0.02  # -2%
        
        # Technical factor adjustments
        rsi = features.get('rsi', 50)
        macd = features.get('macd', 0)
        bb_pos = features.get('bb_position', 0.5)
        vol_regime = features.get('volatility_regime', 1)
        trend_regime = features.get('trend_regime', 1)
        
        # RSI adjustments
        if rsi > 70:  # Overbought
            buy_threshold += 0.015
        elif rsi < 30:  # Oversold
            sell_threshold -= 0.015
        
        # MACD momentum adjustments
        if macd > 0 and features.get('macd_momentum', 0) > 0:  # Strong bullish
            buy_threshold -= 0.01
        elif macd < 0 and features.get('macd_momentum', 0) < 0:  # Strong bearish
            sell_threshold += 0.01
        
        # Bollinger Band adjustments
        if bb_pos > 0.8:  # Near upper band
            buy_threshold += 0.01
        elif bb_pos < 0.2:  # Near lower band
            sell_threshold -= 0.01
        
        # Volatility regime adjustments
        if vol_regime == 2:  # High volatility
            buy_threshold += 0.005
            sell_threshold -= 0.005
        
        # Trend regime adjustments
        if trend_regime == 2:  # Strong bullish trend
            buy_threshold -= 0.005
        elif trend_regime == 0:  # Strong bearish trend
            sell_threshold += 0.005
        
        # Support/Resistance adjustments
        support_dist = features.get('distance_to_support', 0)
        resistance_dist = features.get('distance_to_resistance', 0)
        
        if support_dist < 0.02 and features.get('support_strength', 0) > 2:  # Near strong support
            sell_threshold -= 0.01
        if resistance_dist < 0.02 and features.get('resistance_strength', 0) > 2:  # Near strong resistance
            buy_threshold += 0.01
        
        # Final label determination
        if price_change > buy_threshold:
            return 1  # Buy
        elif price_change < sell_threshold:
            return 0  # Sell
        else:
            return 2  # Hold
    
    def generate_enhanced_chart(self, data, start_idx, window_size=50, save_path=None):
        """Generate enhanced candlestick chart with technical indicators"""
        try:
            end_idx = start_idx + window_size
            if end_idx > len(data):
                return False
                
            chart_data = data.iloc[start_idx:end_idx].copy()
            
            # Ensure data is clean
            chart_data = chart_data.dropna()
            if len(chart_data) < window_size * 0.8:
                return False
            
            # Prepare additional plots
            apds = []
            
            # Add moving averages
            apds.append(mpf.make_addplot(chart_data['SMA_20'], color='blue', width=1))
            apds.append(mpf.make_addplot(chart_data['SMA_50'], color='red', width=1))
            
            # Add Bollinger Bands
            apds.append(mpf.make_addplot(chart_data['BB_Upper'], color='gray', width=0.5, alpha=0.7))
            apds.append(mpf.make_addplot(chart_data['BB_Lower'], color='gray', width=0.5, alpha=0.7))
            
            # Add volume
            volume_panel = mpf.make_addplot(chart_data['Volume'], panel=1, color='lightblue', type='bar')
            apds.append(volume_panel)
            
            # Add RSI
            rsi_panel = mpf.make_addplot(chart_data['RSI'], panel=2, color='purple', ylabel='RSI')
            apds.append(rsi_panel)
            
            # Add MACD
            macd_panel = mpf.make_addplot(chart_data['MACD'], panel=3, color='blue', ylabel='MACD')
            macd_signal_panel = mpf.make_addplot(chart_data['MACD_Signal'], panel=3, color='red')
            apds.extend([macd_panel, macd_signal_panel])
            
            # Create custom style
            mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick={'up':'green', 'down':'red'})
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', y_on_right=False)
            
            # Generate chart
            fig, axes = mpf.plot(
                chart_data,
                type='candle',
                style=s,
                addplot=apds,
                figsize=(12, 10),
                panel_ratios=(3, 1, 1, 1),
                returnfig=True,
                savefig=save_path if save_path else dict(fname='temp.png', dpi=150),
                show_nontrading=False
            )
            
            if save_path:
                plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"    Error generating enhanced chart: {e}")
            return False
    
    def fetch_stock_data(self, symbol, period="2y", max_retries=3):
        """Fetch stock data from Yahoo Finance with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1} for {symbol}...")
                
                # Add random delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
                # Create ticker object and fetch data
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                
                # Validate data
                if data is None or data.empty:
                    print(f"  No data returned for {symbol}")
                    continue
                    
                if len(data) < 100:  # Need more data for technical indicators
                    print(f"  Insufficient data for {symbol}: {len(data)} days")
                    continue
                
                # Check for required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    print(f"  Missing required columns for {symbol}")
                    continue
                
                # Remove any rows with NaN values
                data = data.dropna()
                
                if len(data) < 100:
                    print(f"  Insufficient clean data for {symbol}: {len(data)} days")
                    continue
                
                print(f"  Successfully fetched {len(data)} days of data for {symbol}")
                return data
                
            except Exception as e:
                print(f"  Failed to get ticker '{symbol}' reason: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  {symbol}: No price data found after {max_retries} attempts")
        
        return None
    
    def generate_enhanced_dataset(self, samples_per_stock=50, window_size=50):
        """Generate enhanced dataset with comprehensive technical indicators"""
        labels_data = []
        image_count = 0
        successful_stocks = 0
        
        print("Starting enhanced dataset generation...")
        print(f"Attempting to process {len(self.stocks)} stocks...")
        
        for i, stock in enumerate(self.stocks, 1):
            print(f"\nProcessing {stock} ({i}/{len(self.stocks)})...")
            
            # Fetch stock data
            raw_data = self.fetch_stock_data(stock)
            if raw_data is None or len(raw_data) < window_size + 50:
                print(f"  Skipping {stock} - insufficient data")
                continue
            
            # Calculate enhanced technical indicators
            print(f"  Calculating enhanced technical indicators for {stock}...")
            data = self.calculate_technical_indicators(raw_data)
            
            # Remove rows with NaN values (from technical indicators)
            data = data.dropna()
            
            if len(data) < window_size + 20:
                print(f"  Skipping {stock} - insufficient data after technical calculation")
                continue
            
            successful_stocks += 1
            
            # Generate samples for this stock
            samples_generated = 0
            max_start_idx = len(data) - window_size - 10
            
            # Generate random starting points
            num_samples = min(samples_per_stock, max_start_idx)
            start_indices = np.random.choice(
                range(60, max_start_idx),  # Start after enough data for indicators
                size=num_samples, 
                replace=False
            )
            
            for start_idx in start_indices:
                # Calculate enhanced label
                label, features = self.enhanced_labeling(data, start_idx + window_size - 1)
                if label is None:
                    continue
                
                # Generate image filename
                image_filename = f"enhanced_{stock}_{start_idx}_{image_count}.png"
                image_path = os.path.join(self.chart_dir, image_filename)
                
                # Generate enhanced chart
                success = self.generate_enhanced_chart(
                    data, start_idx, window_size, image_path
                )
                
                if success:
                    # Store enhanced features
                    label_data = {
                        'filename': image_filename,
                        'symbol': stock,
                        'label': label,
                        'start_date': data.index[start_idx].strftime('%Y-%m-%d'),
                        'end_date': data.index[start_idx + window_size - 1].strftime('%Y-%m-%d')
                    }
                    
                    # Add all technical features
                    label_data.update(features)
                    
                    labels_data.append(label_data)
                    samples_generated += 1
                    image_count += 1
                    
                    if samples_generated >= samples_per_stock:
                        break
            
            print(f"  Generated {samples_generated} enhanced samples for {stock}")
        
        # Check if we have any data
        if not labels_data:
            print("\nERROR: No enhanced data was successfully generated!")
            return pd.DataFrame()
        
        # Save labels to CSV
        labels_df = pd.DataFrame(labels_data)
        labels_df.to_csv(self.label_file, index=False)
        
        print(f"\nEnhanced dataset generation complete!")
        print(f"Successfully processed: {successful_stocks}/{len(self.stocks)} stocks")
        print(f"Total enhanced images generated: {len(labels_df)}")
        
        if len(labels_df) > 0:
            print(f"Label distribution:")
            print(labels_df['label'].value_counts())
        
        print(f"Enhanced labels saved to: {self.label_file}")
        
        return labels_df

if __name__ == "__main__":
    print("Enhanced Candlestick Data Generator v2.0")
    print("=" * 50)
    
    generator = EnhancedCandlestickDataGenerator()
    labels_df = generator.generate_enhanced_dataset(samples_per_stock=30, window_size=50)
    
    if not labels_df.empty:
        print("\n" + "="*50)
        print("ENHANCED DATA GENERATION COMPLETE!")
        print("="*50)
        print(f"Enhanced charts saved in: {generator.chart_dir}")
        print(f"Enhanced labels saved in: {generator.label_file}")
    else:
        print("\n" + "="*50)
        print("ENHANCED DATA GENERATION FAILED!")
        print("="*50)