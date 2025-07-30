import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
from enhanced_model_trainer import EnhancedMultiModalCNN, load_enhanced_model_for_inference
from enhanced_data_generator import EnhancedCandlestickDataGenerator
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

class TradingInferenceEngine:
    """Real-time trading inference engine"""
    
    def __init__(self, model_path, device=None, confidence_threshold=0.6):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the trained model
            device: torch device to use
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model, self.checkpoint = load_enhanced_model_for_inference(
            model_path, self.device, use_technical=True
        )
        
        # Load scaler if available
        scaler_path = os.path.join(os.path.dirname(model_path), 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("Loaded feature scaler")
        else:
            print("Warning: No feature scaler found, creating new one")
            self.scaler = StandardScaler()
        
        # Initialize data generator for technical indicators
        self.data_generator = EnhancedCandlestickDataGenerator()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Inference engine initialized on {self.device}")
    
    def fetch_realtime_data(self, symbol, period="60d"):
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval="1d")
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def prepare_technical_features(self, data):
        """Extract and normalize technical features"""
        # Calculate technical indicators
        enhanced_data = self.data_generator.calculate_technical_indicators(data)
        
        # Extract the latest features
        latest_data = enhanced_data.iloc[-1]
        
        feature_names = [
            'rsi', 'macd', 'bb_position', 'stoch_k', 'volume_ratio',
            'volatility', 'distance_to_support', 'distance_to_resistance',
            'vpt', 'bb_squeeze', 'rsi_momentum', 'macd_momentum',
            'support_strength', 'resistance_strength', 'volatility_regime', 'trend_regime'
        ]
        
        features = []
        for feature in feature_names:
            value = latest_data.get(feature, 0)
            if pd.isna(value):
                value = 0
            features.append(value)
        
        # Normalize features
        features = np.array(features).reshape(1, -1)
        
        # Try to use the loaded scaler, otherwise fit a new one
        try:
            normalized_features = self.scaler.transform(features)
        except:
            # If scaler hasn't been fitted, just return normalized values
            normalized_features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return torch.FloatTensor(normalized_features)
    
    def generate_chart_image(self, data, save_path=None):
        """Generate candlestick chart for the last 50 days"""
        try:
            if len(data) < 50:
                chart_data = data
            else:
                chart_data = data.tail(50)
            
            # Use the data generator to create chart
            temp_path = save_path if save_path else "temp_chart.png"
            
            success = self.data_generator.generate_enhanced_chart(
                data, len(data) - len(chart_data), len(chart_data), temp_path
            )
            
            if success:
                return temp_path
            else:
                raise Exception("Failed to generate chart")
                
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None
    
    def predict_single(self, symbol, return_probabilities=False, save_chart=None):
        """
        Make prediction for a single symbol
        
        Args:
            symbol: Stock symbol to predict
            return_probabilities: Whether to return class probabilities
            save_chart: Path to save the generated chart
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Fetch data
            print(f"Fetching data for {symbol}...")
            data = self.fetch_realtime_data(symbol)
            if data is None:
                return None
            
            # Calculate technical indicators
            print("Calculating technical indicators...")
            enhanced_data = self.data_generator.calculate_technical_indicators(data)
            
            # Prepare technical features
            technical_features = self.prepare_technical_features(enhanced_data)
            technical_features = technical_features.to(self.device)
            
            # Generate chart
            print("Generating chart...")
            chart_path = self.generate_chart_image(enhanced_data, save_chart)
            if chart_path is None:
                return None
            
            # Load and preprocess image
            image = Image.open(chart_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            print("Making prediction...")
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor, technical_features)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            # Clean up temporary chart if needed
            if save_chart is None and os.path.exists(chart_path):
                os.remove(chart_path)
            
            # Interpret results
            class_names = ['Sell', 'Buy', 'Hold']
            prediction = class_names[predicted_class]
            
            # Check confidence threshold
            high_confidence = confidence >= self.confidence_threshold
            
            result = {
                'symbol': symbol,
                'prediction': prediction,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'high_confidence': high_confidence,
                'timestamp': datetime.now().isoformat(),
                'current_price': enhanced_data['Close'].iloc[-1],
                'technical_summary': self.get_technical_summary(enhanced_data)
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'Sell': probabilities[0][0].item(),
                    'Buy': probabilities[0][1].item(),
                    'Hold': probabilities[0][2].item()
                }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction for {symbol}: {e}")
            return None
    
    def get_technical_summary(self, data):
        """Get summary of technical indicators"""
        latest = data.iloc[-1]
        
        summary = {
            'rsi': latest.get('RSI', np.nan),
            'macd_signal': 'Bullish' if latest.get('MACD', 0) > latest.get('MACD_Signal', 0) else 'Bearish',
            'bb_position': latest.get('BB_Position', np.nan),
            'trend': 'Bullish' if latest.get('Trend_Regime', 1) == 2 else 'Bearish' if latest.get('Trend_Regime', 1) == 0 else 'Sideways',
            'volatility': 'High' if latest.get('Volatility_Regime', 1) == 2 else 'Low' if latest.get('Volatility_Regime', 1) == 0 else 'Medium'
        }
        
        return summary
    
    def predict_multiple(self, symbols, save_results=True):
        """
        Make predictions for multiple symbols
        
        Args:
            symbols: List of stock symbols
            save_results: Whether to save results to CSV
            
        Returns:
            DataFrame with all predictions
        """
        results = []
        
        print(f"Making predictions for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\nProcessing {symbol} ({i}/{len(symbols)})...")
            
            result = self.predict_single(symbol, return_probabilities=True)
            if result:
                results.append(result)
                print(f"  Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
            else:
                print(f"  Failed to process {symbol}")
        
        if not results:
            print("No successful predictions made")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add probabilities as separate columns
        if 'probabilities' in df.columns:
            prob_df = pd.json_normalize(df['probabilities'])
            prob_df.columns = [f'prob_{col.lower()}' for col in prob_df.columns]
            df = pd.concat([df.drop('probabilities', axis=1), prob_df], axis=1)
        
        # Add technical summary as separate columns
        if 'technical_summary' in df.columns:
            tech_df = pd.json_normalize(df['technical_summary'])
            tech_df.columns = [f'tech_{col}' for col in tech_df.columns]
            df = pd.concat([df.drop('technical_summary', axis=1), tech_df], axis=1)
        
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_predictions_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
        
        return df
    
    def live_monitoring(self, symbols, interval_minutes=30, max_iterations=None):
        """
        Monitor symbols continuously and make predictions
        
        Args:
            symbols: List of symbols to monitor
            interval_minutes: How often to make predictions
            max_iterations: Maximum number of iterations (None for infinite)
        """
        iteration = 0
        
        print(f"Starting live monitoring for {symbols}")
        print(f"Update interval: {interval_minutes} minutes")
        print("Press Ctrl+C to stop")
        
        try:
            while max_iterations is None or iteration < max_iterations:
                print(f"\n{'='*50}")
                print(f"Monitoring iteration {iteration + 1}")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                
                # Make predictions
                results_df = self.predict_multiple(symbols, save_results=True)
                
                if results_df is not None:
                    # Display summary
                    print("\nPrediction Summary:")
                    print("-" * 30)
                    for _, row in results_df.iterrows():
                        conf_flag = "🔥" if row['high_confidence'] else "⚠️"
                        print(f"{conf_flag} {row['symbol']}: {row['prediction']} "
                              f"(Conf: {row['confidence']:.3f}, Price: ${row['current_price']:.2f})")
                
                iteration += 1
                
                if max_iterations is None or iteration < max_iterations:
                    print(f"\nSleeping for {interval_minutes} minutes...")
                    import time
                    time.sleep(interval_minutes * 60)
                    
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error in live monitoring: {e}")

class TradingSignalGenerator:
    """Generate trading signals with risk management"""
    
    def __init__(self, inference_engine, risk_params=None):
        self.inference_engine = inference_engine
        self.risk_params = risk_params or {
            'max_position_size': 0.1,  # Max 10% of portfolio per position
            'stop_loss': 0.05,         # 5% stop loss
            'take_profit': 0.15,       # 15% take profit
            'max_daily_trades': 5,     # Max trades per day
            'min_confidence': 0.7      # Minimum confidence for signals
        }
    
    def generate_signals(self, symbols, portfolio_value=100000):
        """
        Generate trading signals with position sizing
        
        Args:
            symbols: List of symbols to analyze
            portfolio_value: Current portfolio value
            
        Returns:
            List of trading signals
        """
        predictions = self.inference_engine.predict_multiple(symbols)
        if predictions is None:
            return []
        
        signals = []
        
        for _, row in predictions.iterrows():
            if row['confidence'] >= self.risk_params['min_confidence']:
                
                # Calculate position size
                position_size = min(
                    portfolio_value * self.risk_params['max_position_size'],
                    portfolio_value * 0.02 * row['confidence']  # Scale by confidence
                )
                
                signal = {
                    'symbol': row['symbol'],
                    'action': row['prediction'],
                    'confidence': row['confidence'],
                    'current_price': row['current_price'],
                    'position_size': position_size,
                    'stop_loss': row['current_price'] * (1 - self.risk_params['stop_loss']) 
                                if row['prediction'] == 'Buy' 
                                else row['current_price'] * (1 + self.risk_params['stop_loss']),
                    'take_profit': row['current_price'] * (1 + self.risk_params['take_profit']) 
                                  if row['prediction'] == 'Buy' 
                                  else row['current_price'] * (1 - self.risk_params['take_profit']),
                    'timestamp': datetime.now().isoformat(),
                    'technical_summary': row.get('tech_rsi', 'N/A')
                }
                
                signals.append(signal)
        
        return signals

def main():
    """Main function for running inference"""
    print("Trading Inference Engine v2.0")
    print("=" * 40)
    
    # Configuration
    model_path = "enhanced_model/best_enhanced_model.pth"
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using enhanced_model_trainer.py")
        return
    
    try:
        # Initialize inference engine
        engine = TradingInferenceEngine(model_path, confidence_threshold=0.6)
        
        # Test single prediction
        print("\n1. Testing single prediction...")
        result = engine.predict_single("AAPL", return_probabilities=True, save_chart="aapl_chart.png")
        if result:
            print(f"Prediction for AAPL: {result['prediction']} (Confidence: {result['confidence']:.3f})")
            print(f"Technical Summary: {result['technical_summary']}")
        
        # Test multiple predictions
        print(f"\n2. Testing multiple predictions for {test_symbols}...")
        results_df = engine.predict_multiple(test_symbols)
        if results_df is not None:
            print("\nPrediction Results:")
            for _, row in results_df.iterrows():
                print(f"{row['symbol']}: {row['prediction']} "
                      f"(Conf: {row['confidence']:.3f}, Price: ${row['current_price']:.2f})")
        
        # Test signal generation
        print("\n3. Testing signal generation...")
        signal_generator = TradingSignalGenerator(engine)
        signals = signal_generator.generate_signals(test_symbols)
        
        if signals:
            print("Generated Signals:")
            for signal in signals:
                print(f"  {signal['symbol']}: {signal['action']} at ${signal['current_price']:.2f} "
                      f"(Size: ${signal['position_size']:.0f}, Conf: {signal['confidence']:.3f})")
        else:
            print("No high-confidence signals generated")
        
        print("\n" + "="*40)
        print("Inference testing complete!")
        print("="*40)
        
        # Ask if user wants to start live monitoring
        response = input("\nStart live monitoring? (y/n): ").lower().strip()
        if response == 'y':
            engine.live_monitoring(test_symbols, interval_minutes=5, max_iterations=3)
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()