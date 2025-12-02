import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import Counter, defaultdict
import os
import math
import json
import hashlib
from quant_excel_loader import load_results_excel

# ========== AGGRESSIVE TENSORFLOW SUPPRESSION ==========
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Deep Learning - Import with extreme suppression
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
import logging
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Advanced ML
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Global warnings suppression
warnings.filterwarnings('ignore')

class UltimatePredictorPro:
    def __init__(self):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.performance_log = "ultimate_performance.csv"
        self.models = {}
        self.lstm_models = {}  # Cache for LSTM models - PHASE 1
        
        # Advanced Pattern Packs - PHASE 2
        self.initialize_pattern_packs()
        self.initialize_cross_script_tracking()
        self.initialize_performance_tracking()
        
        # Create output directory - PHASE 3 (FIXED PATH)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "predictions", "deepseek_scr6")
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_file = os.path.join(self.output_dir, "scr6_cache.json")
        self.cache_wide_file = os.path.join(self.output_dir, "scr6_predictions_latest.xlsx")
        self.cache_detailed_file = os.path.join(self.output_dir, "scr6_predictions_latest_detailed.xlsx")
        print(f"📁 Output directory: {self.output_dir}")
    
    def initialize_cross_script_tracking(self):
        """Initialize cross-script pattern learning"""
        # FIX: Use proper defaultdict initialization
        self.cross_script_patterns = defaultdict(lambda: defaultdict(int))
        self.script_effectiveness = defaultdict(lambda: defaultdict(float))
        self.cross_date_patterns = defaultdict(lambda: defaultdict(int))
        
        print("🔄 Cross-script pattern tracking initialized")
    
    def initialize_pattern_packs(self):
        """Initialize advanced pattern detection packs - PHASE 2"""
        # 3-digit pattern families
        self.pattern_3digit = {
            '123/123': self.generate_digit_pattern([1,2,3]),
            '234/234': self.generate_digit_pattern([2,3,4]),
            '345/345': self.generate_digit_pattern([3,4,5]),
            '456/456': self.generate_digit_pattern([4,5,6]),
            '567/567': self.generate_digit_pattern([5,6,7]),
            '678/678': self.generate_digit_pattern([6,7,8]),
            '789/789': self.generate_digit_pattern([7,8,9]),
            '890/890': self.generate_digit_pattern([8,9,0]),
            '901/901': self.generate_digit_pattern([9,0,1]),
            '012/012': self.generate_digit_pattern([0,1,2])
        }
        
        # 4-digit pattern families
        self.pattern_4digit = {
            '1234/1234': self.generate_digit_pattern([1,2,3,4]),
            '2345/2345': self.generate_digit_pattern([2,3,4,5]),
            '3456/3456': self.generate_digit_pattern([3,4,5,6]),
            '4567/4567': self.generate_digit_pattern([4,5,6,7]),
            '5678/5678': self.generate_digit_pattern([5,6,7,8]),
            '6789/6789': self.generate_digit_pattern([6,7,8,9]),
            '7890/7890': self.generate_digit_pattern([7,8,9,0]),
            '8901/8901': self.generate_digit_pattern([8,9,0,1]),
            '9012/9012': self.generate_digit_pattern([9,0,1,2]),
            '0123/0123': self.generate_digit_pattern([0,1,2,3])
        }
        
        # 6-digit pattern families (including 164950/164950)
        self.pattern_6digit = {
            '164950/164950': self.generate_digit_pattern([1,6,4,9,5,0]),
            '123456/123456': self.generate_digit_pattern([1,2,3,4,5,6]),
            '234567/234567': self.generate_digit_pattern([2,3,4,5,6,7]),
            '345678/345678': self.generate_digit_pattern([3,4,5,6,7,8]),
            '456789/456789': self.generate_digit_pattern([4,5,6,7,8,9]),
            '567890/567890': self.generate_digit_pattern([5,6,7,8,9,0])
        }
        
        # S40 Pattern Pack (Very Important)
        self.S40_numbers = {
            0, 6, 7, 9, 15, 16, 18, 19, 24, 25, 27, 28, 33, 34, 36, 37,
            42, 43, 45, 46, 51, 52, 54, 55, 60, 61, 63, 64, 70, 72, 73,
            79, 81, 82, 88, 89, 90, 91, 97, 98
        }
        
        # Cross-slot pattern tracking - FIX: Use proper defaultdict
        self.cross_slot_patterns = defaultdict(lambda: defaultdict(int))
        self.pattern_performance = defaultdict(lambda: defaultdict(list))
        
        print("✅ Advanced pattern packs initialized:")
        print(f"   - 3-digit patterns: {len(self.pattern_3digit)} families")
        print(f"   - 4-digit patterns: {len(self.pattern_4digit)} families") 
        print(f"   - 6-digit patterns: {len(self.pattern_6digit)} families")
        print(f"   - S40 numbers: {len(self.S40_numbers)} special numbers")
        print(f"   - 164950/164950 pattern: {len(self.pattern_6digit['164950/164950'])} numbers")
    
    def generate_digit_pattern(self, digits):
        """Generate number patterns from digit sets - PHASE 2"""
        pattern = set()
        for tens in digits:
            for ones in digits:
                number = tens * 10 + ones
                if 0 <= number <= 99:
                    pattern.add(number)
        return pattern
    
    def initialize_performance_tracking(self):
        """Initialize performance tracking file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        perf_dir = os.path.join(script_dir, "logs", "performance")
        os.makedirs(perf_dir, exist_ok=True)
        self.performance_log = os.path.join(perf_dir, "ultimate_performance.csv")
        
        if not os.path.exists(self.performance_log):
            performance_df = pd.DataFrame(columns=[
                'prediction_date', 'slot', 'predicted_numbers', 
                'actual_number', 'hit_rank', 'hit_count', 'accuracy',
                'script_name', 'cross_script_hits', 'cross_date_hits'
            ])
            performance_df.to_csv(self.performance_log, index=False)

    def load_updated_data(self, file_path):
        """Load updated data via canonical quant_excel_loader."""
        try:
            print("📂 SCR6: Loading updated data via quant_excel_loader.")
            df = load_results_excel(file_path)
            if df is None or len(df) == 0:
                print("❌ SCR6: No valid data returned from quant_excel_loader")
                return None

            latest_date = df['date'].max()
            print(f"✅ SCR6: Successfully loaded {len(df)} records")
            print(f"📅 SCR6: Data range: {df['date'].min().strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
            print(f"🕐 SCR6: Latest data point: {latest_date.strftime('%Y-%m-%d')}")
            return df

        except Exception as e:
            print(f"❌ SCR6: Error loading data from {file_path}: {e}")
            return None

    def compute_data_signature(self, df):
        """Compute a lightweight signature of the dataset to drive disk caching."""
        if df is None or len(df) == 0:
            return {}

        if not pd.api.types.is_datetime64_any_dtype(df.get("date")):
            df["date"] = pd.to_datetime(df["date"])

        last_date = df["date"].max().strftime("%Y-%m-%d")
        slot_counts = {str(k): int(v) for k, v in df["slot"].value_counts().to_dict().items()}
        checksum_basis = "".join(df["date"].dt.strftime("%Y-%m-%d"))
        checksum = hashlib.sha256(checksum_basis.encode()).hexdigest()[:16]

        return {
            "last_date": last_date,
            "rows": int(len(df)),
            "slot_counts": slot_counts,
            "date_checksum": checksum,
        }

    def load_cache_meta(self):
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def is_cache_hit(self, signature, meta):
        return (
            bool(signature)
            and bool(meta)
            and meta.get("signature") == signature
            and os.path.exists(self.cache_wide_file)
            and os.path.exists(self.cache_detailed_file)
        )

    def persist_cache(self, signature, predictions_df, wide_df):
        try:
            predictions_df.to_excel(self.cache_detailed_file, index=False)
            wide_df.to_excel(self.cache_wide_file, index=False)
            cache_payload = {
                "signature": signature,
                "wide_file": self.cache_wide_file,
                "detailed_file": self.cache_detailed_file,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self.cache_file, "w") as f:
                json.dump(cache_payload, f, indent=2)
            print(
                f"💾 SCR6 cache rebuilt for current dataset – signature {signature.get('last_date', '?')} / {signature.get('rows', 0)} rows."
            )
        except Exception as e:
            print(f"⚠️ SCR6: Failed to persist cache due to {e}")

    def clean_number(self, x):
        """Convert to 2-digit number"""
        try:
            s = str(x).strip()
            digits = ''.join([c for c in s if c.isdigit()])
            if not digits:
                return None
            num = int(digits)
            return num % 100
        except:
            return None

    # ==================== ADVANCED FEATURE ENGINEERING ====================
    def create_advanced_features(self, numbers):
        """Create comprehensive features for ML models"""
        if len(numbers) < 30:
            return None, None
            
        features = []
        targets = []
        
        lookback = 30
        for i in range(lookback, len(numbers)):
            # Recent numbers as features
            recent = numbers[i-lookback:i]
            
            # Statistical features - 6 features
            feature_row = [
                np.mean(recent), np.std(recent), np.min(recent), np.max(recent),
                np.median(recent), len(set(recent)),  # uniqueness
            ]
            
            # Rolling statistics - 9 features (3 windows * 3 stats)
            for window in [5, 10, 15]:
                if len(recent) >= window:
                    feature_row.extend([
                        np.mean(recent[-window:]),
                        np.std(recent[-window:]),
                        np.median(recent[-window:])
                    ])
                else:
                    feature_row.extend([0, 0, 0])
            
            # Digit features - 4 features
            tens_digits = [n // 10 for n in recent]
            ones_digits = [n % 10 for n in recent]
            feature_row.extend([
                np.mean(tens_digits), np.std(tens_digits),
                np.mean(ones_digits), np.std(ones_digits)
            ])
            
            # Pattern features - 8 features
            pattern_features = self.extract_pattern_features(recent)
            feature_row.extend(pattern_features)
            
            # Total features: 6 + 9 + 4 + 8 = 27 features
            features.append(feature_row)
            targets.append(numbers[i])
        
        return np.array(features), np.array(targets)

    def extract_pattern_features(self, numbers):
        """Extract advanced pattern features - PHASE 2"""
        features = []
        
        # S40 frequency
        s40_count = sum(1 for n in numbers if n in self.S40_numbers)
        features.append(s40_count / len(numbers))
        
        # 3-digit pattern frequencies (top 2)
        pattern_3_counts = []
        for pattern_name, pattern_set in self.pattern_3digit.items():
            count = sum(1 for n in numbers if n in pattern_set)
            pattern_3_counts.append(count / len(numbers))
        features.extend(pattern_3_counts[:2])
        
        # 4-digit pattern frequencies (top 2)
        pattern_4_counts = []
        for pattern_name, pattern_set in self.pattern_4digit.items():
            count = sum(1 for n in numbers if n in pattern_set)
            pattern_4_counts.append(count / len(numbers))
        features.extend(pattern_4_counts[:2])
        
        # 6-digit pattern frequencies (top 2)
        pattern_6_counts = []
        for pattern_name, pattern_set in self.pattern_6digit.items():
            count = sum(1 for n in numbers if n in pattern_set)
            pattern_6_counts.append(count / len(numbers))
        features.extend(pattern_6_counts[:2])
        
        # Cross-slot pattern indicator
        features.extend([0, 0])  # Will be populated in cross-slot analysis
        
        return features

    # ==================== CROSS-SCRIPT PATTERN LEARNING ====================
    def analyze_cross_script_patterns(self, df):
        """Analyze patterns across different scripts and dates - OPTIMIZED"""
        print("🔄 Analyzing cross-script patterns...")
        
        # Load prediction files from other scripts
        script_predictions = self.load_other_script_predictions()
        
        if not script_predictions:
            print("   ⚠️  No other script predictions found for cross-analysis")
            return
        
        # Analyze cross-script effectiveness
        analysis_count = 0
        for script_name, predictions_df in script_predictions.items():
            if predictions_df is None or len(predictions_df) == 0:
                continue
                
            for _, pred_row in predictions_df.iterrows():
                if analysis_count > 1000:  # Limit analysis for performance
                    break
                    
                pred_date = pred_row.get('date')
                pred_slot = pred_row.get('slot')
                pred_numbers_str = pred_row.get('number', '')
                
                if not all([pred_date, pred_slot, pred_numbers_str]):
                    continue
                
                try:
                    pred_numbers = [int(x.strip()) for x in str(pred_numbers_str).split(',') if x.strip().isdigit()]
                    # Check if these predictions hit in other slots/dates
                    self.check_cross_script_hits(script_name, pred_date, pred_slot, pred_numbers, df)
                    analysis_count += 1
                except (ValueError, AttributeError):
                    continue
        
        # Print cross-script effectiveness
        if self.script_effectiveness:
            print("📊 Cross-script effectiveness summary:")
            for script_name, slot_data in list(self.script_effectiveness.items())[:3]:
                for slot, effectiveness in list(slot_data.items())[:2]:
                    print(f"   {script_name} -> {slot}: {effectiveness:.3f}")
        else:
            print("   📊 No cross-script patterns found")

    def load_other_script_predictions(self):
        """Load prediction files from other scripts - OPTIMIZED"""
        script_predictions = {}
        script_folders = [
            'deepseek_scr1', 'deepseek_scr2', 'deepseek_scr3', 
            'deepseek_scr4', 'deepseek_scr5', 'deepseek_scr7', 
            'deepseek_scr8', 'deepseek_scr9'
        ]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        for script_folder in script_folders:
            try:
                pred_path = os.path.join(script_dir, "predictions", script_folder)
                if os.path.exists(pred_path):
                    # Find the latest prediction file
                    pred_files = [f for f in os.listdir(pred_path) 
                                if f.startswith('ultimate_') and f.endswith('.xlsx')]
                    if pred_files:
                        latest_file = max(pred_files)  # Gets the most recent by name
                        full_path = os.path.join(pred_path, latest_file)
                        predictions_df = pd.read_excel(full_path)
                        script_predictions[script_folder] = predictions_df
                        print(f"   ✅ Loaded predictions from {script_folder}")
            except Exception as e:
                continue  # Silent fail for missing scripts
        
        return script_predictions

    def check_cross_script_hits(self, script_name, pred_date, pred_slot, pred_numbers, df):
        """Check if script predictions hit in other slots/dates"""
        # Convert prediction date to datetime
        try:
            pred_datetime = pd.to_datetime(pred_date)
        except:
            return
        
        # Check same date different slots
        same_date_data = df[df['date'] == pred_datetime]
        for _, actual_row in same_date_data.iterrows():
            actual_slot = self.slot_names[actual_row['slot']]
            actual_number = actual_row['number']
            
            if actual_slot != pred_slot and actual_number in pred_numbers:
                # Cross-slot hit found!
                pattern_key = f"{script_name}_{pred_slot}→{actual_slot}"
                self.cross_script_patterns[pattern_key][actual_number] += 1
                
                # Update script effectiveness
                self.script_effectiveness[script_name][f"cross_{actual_slot}"] += 0.1
        
        # Check previous/next dates same slot
        prev_date = pred_datetime - timedelta(days=1)
        next_date = pred_datetime + timedelta(days=1)
        
        for check_date in [prev_date, next_date]:
            date_data = df[df['date'] == check_date]
            slot_data = date_data[date_data['slot'] == list(self.slot_names.keys())[list(self.slot_names.values()).index(pred_slot)]]
            
            if not slot_data.empty:
                actual_number = slot_data['number'].iloc[0]
                if actual_number in pred_numbers:
                    # Cross-date hit found!
                    date_diff = (check_date - pred_datetime).days
                    direction = "prev" if date_diff < 0 else "next"
                    pattern_key = f"{script_name}_{pred_slot}_{direction}day"
                    self.cross_date_patterns[pattern_key][actual_number] += 1
                    
                    # Update script effectiveness
                    self.script_effectiveness[script_name][f"date_{direction}"] += 0.1

    def get_cross_script_boost(self, number, slot_name):
        """Get cross-script pattern boost for a number"""
        boost = 0.0
        
        # Check cross-slot patterns
        for pattern_key, number_counts in self.cross_script_patterns.items():
            if number in number_counts:
                # Example: "scr3_FRBD→DSWR" pattern
                parts = pattern_key.split('_')
                if len(parts) >= 2:
                    source_script = parts[0]
                    slot_pattern = parts[1]
                    if slot_name in slot_pattern:
                        boost += number_counts[number] * 0.2
        
        # Check cross-date patterns
        for pattern_key, number_counts in self.cross_date_patterns.items():
            if number in number_counts:
                boost += number_counts[number] * 0.15
        
        return min(boost, 1.0)  # Cap the boost

    # ==================== REAL LSTM MODEL (FIXED) ====================
    def build_lstm_model(self, lookback=30):
        """Build and compile LSTM model - FIXED to prevent retracing - PHASE 1"""
        if lookback in self.lstm_models:
            return self.lstm_models[lookback]
            
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(lookback, 1)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Cache the model - PHASE 1
        self.lstm_models[lookback] = model
        return model

    def lstm_prediction(self, numbers, top_k=5):
        """Real LSTM prediction with sequence learning - FIXED retracing - PHASE 1"""
        if len(numbers) < 60:
            return []
            
        try:
            # Prepare data for LSTM
            lookback = 30
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(numbers).reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build and train model (using cached version) - PHASE 1
            model = self.build_lstm_model(lookback)
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,  # Reduced for performance
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Predict next value
            last_sequence = scaled_data[-lookback:]
            last_sequence = last_sequence.reshape(1, lookback, 1)
            next_pred_scaled = model.predict(last_sequence, verbose=0)
            next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
            next_pred = int(np.clip(np.round(next_pred), 0, 99))
            
            # Generate top predictions around LSTM prediction with pattern boost
            base_predictions = []
            
            for offset in range(-top_k//2, top_k//2 + 1):
                pred_num = (next_pred + offset) % 100
                if pred_num not in base_predictions:
                    base_predictions.append(pred_num)
            
            # Apply pattern boosting - PHASE 2
            boosted_predictions = self.apply_pattern_boosting(base_predictions, numbers)
            
            return boosted_predictions[:top_k]
            
        except Exception as e:
            return []

    def apply_pattern_boosting(self, base_predictions, historical_numbers):
        """Apply pattern-based boosting to predictions - PHASE 2"""
        scored_predictions = []
        
        for num in base_predictions:
            score = 1.0  # Base score
            
            # S40 Boost - PHASE 2
            if num in self.S40_numbers:
                score += 0.5
            
            # 3-digit pattern boost - PHASE 2
            for pattern_name, pattern_set in self.pattern_3digit.items():
                if num in pattern_set:
                    score += 0.3
                    break
            
            # 4-digit pattern boost - PHASE 2
            for pattern_name, pattern_set in self.pattern_4digit.items():
                if num in pattern_set:
                    score += 0.4
                    break
            
            # 6-digit pattern boost (including 164950/164950) - PHASE 2
            for pattern_name, pattern_set in self.pattern_6digit.items():
                if num in pattern_set:
                    score += 0.5  # Higher boost for 6-digit patterns
                    break
            
            # Cross-slot pattern boost
            cross_boost = self.get_cross_slot_boost(num, historical_numbers)
            score += cross_boost
            
            # Cross-script pattern boost
            script_boost = self.get_cross_script_boost(num, "CURRENT_SLOT")  # Will be set per slot
            score += script_boost
            
            scored_predictions.append((num, score))
        
        # Sort by boosted score
        scored_predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, score in scored_predictions]

    def get_cross_slot_boost(self, number, historical_numbers):
        """Calculate cross-slot pattern boost"""
        if len(historical_numbers) < 10:
            return 0.0
            
        # Simple cross-slot boost based on recent appearances
        recent = historical_numbers[-10:]
        if number in recent:
            return 0.2
        return 0.0

    # ==================== XGBOOST PREDICTION ====================
    def xgboost_prediction(self, numbers, top_k=5):
        """XGBoost prediction with advanced features"""
        if len(numbers) < 50:
            return []
            
        try:
            features, targets = self.create_advanced_features(numbers)
            if features is None or len(features) == 0:
                return []
            
            # Split data
            split_idx = int(0.8 * len(features))
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            # Train XGBoost
            xgb_model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8
            )
            
            xgb_model.fit(X_train, y_train)
            
            # Create feature for prediction (most recent data)
            recent_features = self.create_prediction_features(numbers)
            if recent_features is None:
                return []
                
            # Ensure feature count matches
            if len(recent_features) != X_train.shape[1]:
                return []
            
            # Predict
            prediction = xgb_model.predict([recent_features])[0]
            prediction = int(np.clip(np.round(prediction), 0, 99))
            
            # Generate variations with pattern boost - PHASE 2
            base_predictions = []
            for offset in range(-top_k//2, top_k//2 + 1):
                pred_num = (prediction + offset) % 100
                if pred_num not in base_predictions:
                    base_predictions.append(pred_num)
            
            boosted_predictions = self.apply_pattern_boosting(base_predictions, numbers)
            return boosted_predictions[:top_k]
            
        except Exception as e:
            return []

    def create_prediction_features(self, numbers):
        """Create features for prediction from recent data"""
        if len(numbers) < 30:
            return None
            
        recent = numbers[-30:]
        
        # Statistical features - 6 features
        features = [
            np.mean(recent), np.std(recent), np.min(recent), np.max(recent),
            np.median(recent), len(set(recent)),
        ]
        
        # Rolling stats - 9 features (3 windows * 3 stats)
        for window in [5, 10, 15]:
            if len(recent) >= window:
                features.extend([
                    np.mean(recent[-window:]),
                    np.std(recent[-window:]),
                    np.median(recent[-window:])
                ])
            else:
                features.extend([0, 0, 0])
        
        # Digit features - 4 features
        tens_digits = [n // 10 for n in recent]
        ones_digits = [n % 10 for n in recent]
        features.extend([
            np.mean(tens_digits), np.std(tens_digits),
            np.mean(ones_digits), np.std(ones_digits)
        ])
        
        # Pattern features - 8 features - PHASE 2
        pattern_features = self.extract_pattern_features(recent)
        features.extend(pattern_features)
        
        return features

    # ==================== LIGHTGBM PREDICTION ====================
    def lightgbm_prediction(self, numbers, top_k=5):
        """LightGBM prediction with pattern integration - PHASE 2"""
        if len(numbers) < 50:
            return []
            
        try:
            features, targets = self.create_advanced_features(numbers)
            if features is None or len(features) == 0:
                return []
            
            split_idx = int(0.8 * len(features))
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            lgb_model = LGBMRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=7,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            )
            
            lgb_model.fit(X_train, y_train)
            
            recent_features = self.create_prediction_features(numbers)
            if recent_features is None:
                return []
                
            if len(recent_features) != X_train.shape[1]:
                return []
            
            prediction = lgb_model.predict([recent_features])[0]
            prediction = int(np.clip(np.round(prediction), 0, 99))
            
            base_predictions = []
            for offset in range(-top_k//2, top_k//2 + 1):
                pred_num = (prediction + offset) % 100
                if pred_num not in base_predictions:
                    base_predictions.append(pred_num)
            
            boosted_predictions = self.apply_pattern_boosting(base_predictions, numbers)
            return boosted_predictions[:top_k]
            
        except Exception as e:
            return []

    # ==================== NEURAL NETWORK PREDICTION ====================
    def neural_network_prediction(self, numbers, top_k=5):
        """Neural Network prediction with patterns - PHASE 2"""
        if len(numbers) < 50:
            return []
            
        try:
            features, targets = self.create_advanced_features(numbers)
            if features is None or len(features) == 0:
                return []
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            split_idx = int(0.8 * len(features))
            X_train, X_test = features_scaled[:split_idx], features_scaled[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            nn_model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            
            nn_model.fit(X_train, y_train)
            
            recent_features = self.create_prediction_features(numbers)
            if recent_features is None:
                return []
                
            if len(recent_features) != X_train.shape[1]:
                return []
                
            recent_scaled = scaler.transform([recent_features])
            prediction = nn_model.predict(recent_scaled)[0]
            prediction = int(np.clip(np.round(prediction), 0, 99))
            
            base_predictions = []
            for offset in range(-top_k//2, top_k//2 + 1):
                pred_num = (prediction + offset) % 100
                if pred_num not in base_predictions:
                    base_predictions.append(pred_num)
            
            boosted_predictions = self.apply_pattern_boosting(base_predictions, numbers)
            return boosted_predictions[:top_k]
            
        except Exception as e:
            return []

    # ==================== ADVANCED PATTERN DETECTION ====================
    def detect_cross_slot_patterns(self, df):
        """Detect patterns across different slots and dates"""
        print("🔄 Analyzing cross-slot patterns...")
        
        # Group by date and collect all numbers
        date_slot_data = defaultdict(dict)
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            slot_name = self.slot_names[row['slot']]
            date_slot_data[date_str][slot_name] = row['number']
        
        # Convert to list of dates for analysis
        dates = sorted(date_slot_data.keys())
        
        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i-1]
            
            current_data = date_slot_data[current_date]
            prev_data = date_slot_data[prev_date]
            
            # Check for numbers moving between slots
            for curr_slot, curr_number in current_data.items():
                for prev_slot, prev_number in prev_data.items():
                    if curr_number == prev_number:
                        pattern_key = f"{prev_slot}→{curr_slot}"
                        # FIX: Direct integer increment for defaultdict(int)
                        self.cross_slot_patterns[pattern_key][curr_number] += 1
        
        # Print pattern summary
        print("📊 Cross-slot pattern summary:")
        pattern_count = 0
        for pattern, numbers in self.cross_slot_patterns.items():
            if numbers and pattern_count < 5:
                # Convert to Counter to get most common
                counter = Counter(numbers)
                top_number = counter.most_common(1)[0] if counter else ("None", 0)
                print(f"   {pattern}: {top_number[0]} (count: {top_number[1]})")
                pattern_count += 1

    def analyze_pattern_performance(self, df):
        """Analyze performance of different pattern packs - PHASE 2"""
        print("📈 Analyzing pattern pack performance...")
        
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            numbers = slot_data['number'].tolist()
            
            if len(numbers) < 10:
                continue
            
            # Analyze S40 performance - PHASE 2
            s40_hits = sum(1 for n in numbers if n in self.S40_numbers)
            s40_percentage = (s40_hits / len(numbers)) * 100
            
            # Analyze 6-digit pattern performance - PHASE 2
            pattern_6_performance = {}
            for pattern_name, pattern_set in self.pattern_6digit.items():
                hits = sum(1 for n in numbers if n in pattern_set)
                percentage = (hits / len(numbers)) * 100
                pattern_6_performance[pattern_name] = percentage
            
            # Store performance data
            self.pattern_performance[slot_name]['S40'] = s40_percentage
            self.pattern_performance[slot_name]['6digit'] = pattern_6_performance
            
            print(f"   {slot_name}: S40={s40_percentage:.1f}%, 164950={pattern_6_performance.get('164950/164950', 0):.1f}%")

    # ==================== ENSEMBLE STRATEGIES ====================
    def advanced_ensemble(self, numbers, top_k=10):
        """10-Strategy Ultimate Ensemble with Pattern Integration"""
        if len(numbers) < 30:
            return self.fallback_prediction(numbers, top_k)
        
        strategies = {}
        
        # 1. Real LSTM (Fixed) - PHASE 1
        lstm_pred = self.lstm_prediction(numbers, top_k)
        if lstm_pred:
            strategies['lstm'] = (lstm_pred, 0.15)
        
        # 2. XGBoost
        xgb_pred = self.xgboost_prediction(numbers, top_k)
        if xgb_pred:
            strategies['xgboost'] = (xgb_pred, 0.15)
        
        # 3. LightGBM
        lgb_pred = self.lightgbm_prediction(numbers, top_k)
        if lgb_pred:
            strategies['lightgbm'] = (lgb_pred, 0.12)
        
        # 4. Neural Network
        nn_pred = self.neural_network_prediction(numbers, top_k)
        if nn_pred:
            strategies['neural_net'] = (nn_pred, 0.10)
        
        # 5. Bayesian Probability
        bayesian_pred = self.bayesian_probability(numbers, top_k)
        strategies['bayesian'] = (bayesian_pred, 0.10)
        
        # 6. Gap Analysis
        gap_pred = self.confidence_gap_analysis(numbers, top_k)
        strategies['gap_analysis'] = (gap_pred, 0.10)
        
        # 7. Pattern Mining (Enhanced)
        pattern_pred = self.advanced_pattern_mining(numbers, top_k)
        strategies['patterns'] = (pattern_pred, 0.08)
        
        # 8. Frequency Momentum
        momentum_pred = self.frequency_momentum(numbers, top_k)
        strategies['momentum'] = (momentum_pred, 0.08)
        
        # 9. Markov Chains
        markov_pred = self.multi_step_markov(numbers, top_k)
        strategies['markov'] = (markov_pred, 0.06)
        
        # 10. Statistical Forest
        rf_pred = self.statistical_forest(numbers, top_k)
        strategies['statistical'] = (rf_pred, 0.06)
        
        # Combine all strategies
        final_scores = self.calibrate_predictions(strategies, numbers)
        
        # Apply intelligent filtering with pattern awareness - PHASE 2
        final_predictions = self.apply_intelligent_filter(final_scores, top_k)
        
        return final_predictions

    # ==================== CORE STRATEGIES (PRESERVED) ====================
    def bayesian_probability(self, numbers, top_k):
        """Bayesian probability with multiple time windows"""
        windows = [30, 60, 90, 180]
        alpha = 1.0
        
        combined_probs = {}
        
        for window in windows:
            if len(numbers) >= window:
                recent = numbers[-window:]
                freq = Counter(recent)
                total = len(recent)
                
                for num in range(100):
                    count = freq.get(num, 0)
                    prob = (count + alpha) / (total + 100 * alpha)
                    combined_probs[num] = combined_probs.get(num, 0) + prob
        
        if not combined_probs:
            freq = Counter(numbers)
            total = len(numbers)
            for num in range(100):
                count = freq.get(num, 0)
                combined_probs[num] = (count + alpha) / (total + 100 * alpha)
        
        sorted_probs = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
        return [num for num, prob in sorted_probs[:top_k]]

    def confidence_gap_analysis(self, numbers, top_k):
        """Gap analysis with confidence intervals"""
        positions = {}
        for i, num in enumerate(numbers):
            if num not in positions:
                positions[num] = []
            positions[num].append(i)
        
        gap_scores = {}
        current_idx = len(numbers) - 1
        
        for num in range(100):
            if num in positions and len(positions[num]) > 2:
                gaps = [positions[num][i] - positions[num][i-1] for i in range(1, len(positions[num]))]
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                current_gap = current_idx - positions[num][-1]
                
                if current_gap > avg_gap:
                    confidence = min((current_gap - avg_gap) / (std_gap + 1), 3.0) / 3.0
                else:
                    confidence = 0.1
                    
                gap_scores[num] = confidence
            else:
                gap_scores[num] = 0.5
        
        due_numbers = sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in due_numbers[:top_k]]

    def advanced_pattern_mining(self, numbers, top_k):
        """Advanced pattern mining with cross-slot integration"""
        sequences = defaultdict(list)
        
        for length in [2, 3, 4]:
            for i in range(len(numbers) - length):
                seq = tuple(numbers[i:i+length])
                next_val = numbers[i+length]
                sequences[seq].append(next_val)
        
        predictions = []
        for length in [4, 3, 2]:
            if len(numbers) >= length:
                recent_seq = tuple(numbers[-length:])
                if recent_seq in sequences:
                    next_vals = sequences[recent_seq]
                    counter = Counter(next_vals)
                    predictions.extend([num for num, count in counter.most_common(3)])
        
        if not predictions:
            freq = Counter(numbers[-30:])
            predictions = [num for num, count in freq.most_common(top_k)]
        
        return predictions[:top_k]

    def frequency_momentum(self, numbers, top_k):
        """Frequency momentum tracking"""
        if len(numbers) < 20:
            return self.fallback_prediction(numbers, top_k)
        
        recent_window = min(20, len(numbers) // 2)
        historical_window = min(60, len(numbers))
        
        if historical_window <= recent_window:
            return self.fallback_prediction(numbers, top_k)
        
        recent = numbers[-recent_window:]
        historical = numbers[-historical_window:-recent_window]
        
        recent_freq = Counter(recent)
        historical_freq = Counter(historical)
        
        momentum_scores = {}
        
        for num in range(100):
            recent_count = recent_freq.get(num, 0)
            historical_count = historical_freq.get(num, 0)
            
            if historical_count > 0:
                momentum = recent_count / recent_window - historical_count / len(historical)
            else:
                momentum = recent_count / recent_window
                
            momentum_scores[num] = max(momentum, 0)
        
        rising_numbers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in rising_numbers[:top_k]]

    def multi_step_markov(self, numbers, top_k):
        """Multi-step Markov chain"""
        if len(numbers) < 10:
            return self.fallback_prediction(numbers, top_k)
        
        transitions = {}
        for i in range(1, len(numbers)):
            prev = numbers[i-1]
            curr = numbers[i]
            
            if prev not in transitions:
                transitions[prev] = {}
            transitions[prev][curr] = transitions[prev].get(curr, 0) + 1
        
        predictions = []
        current = numbers[-1]
        
        for step in range(min(5, top_k)):
            if current in transitions:
                next_probs = transitions[current]
                next_num = max(next_probs.items(), key=lambda x: x[1])[0]
                predictions.append(next_num)
                current = next_num
            else:
                break
        
        if len(predictions) < top_k:
            freq = Counter(numbers[-20:])
            additional = [num for num, count in freq.most_common(top_k) if num not in predictions]
            predictions.extend(additional[:top_k - len(predictions)])
        
        return predictions[:top_k]

    def statistical_forest(self, numbers, top_k):
        """Statistical forest simulation"""
        if len(numbers) < 15:
            return self.fallback_prediction(numbers, top_k)
        
        features = {}
        
        recent_freq = Counter(numbers[-15:])
        for num, count in recent_freq.items():
            features[num] = features.get(num, 0) + count * 0.3
        
        last_positions = {}
        for i, num in enumerate(numbers[-30:]):
            last_positions[num] = i
        for num, pos in last_positions.items():
            features[num] = features.get(num, 0) + (30 - pos) * 0.2
        
        last_seen = {}
        for i, num in enumerate(numbers):
            last_seen[num] = i
        current_idx = len(numbers) - 1
        for num in range(100):
            gap = current_idx - last_seen.get(num, 0)
            features[num] = features.get(num, 0) + min(gap, 20) * 0.1
        
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in sorted_features[:top_k]]

    def calibrate_predictions(self, strategies, numbers):
        """Calibrate prediction scores"""
        final_scores = defaultdict(float)
        
        for strategy_name, (predictions, base_weight) in strategies.items():
            reliability = self.assess_strategy_reliability(strategy_name, numbers)
            adjusted_weight = base_weight * reliability
            
            for rank, num in enumerate(predictions):
                position_weight = (len(predictions) - rank) / len(predictions)
                final_scores[num] += adjusted_weight * position_weight
        
        return final_scores

    def assess_strategy_reliability(self, strategy_name, numbers):
        """Assess strategy reliability"""
        if len(numbers) < 10:
            return 0.5
        
        data_diversity = len(set(numbers)) / 100
        data_volume = min(len(numbers) / 100, 1.0)
        
        if strategy_name in ['lstm', 'xgboost', 'lightgbm']:
            return 0.9 * data_volume
        elif strategy_name == 'neural_net':
            return 0.8 * data_volume
        elif strategy_name == 'bayesian':
            return 0.9 * data_volume
        elif strategy_name == 'gap_analysis':
            return 0.8 * data_diversity
        else:
            return 0.7
        
        return 0.5

    def apply_intelligent_filter(self, scores, top_k):
        """Intelligent filtering for balanced predictions with pattern awareness - PHASE 2"""
        if not scores:
            return list(range(top_k))
        
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [num for num, score in sorted_predictions[:top_k * 2]]
        
        ranges = {
            'low': [n for n in candidates if 0 <= n <= 33],
            'medium': [n for n in candidates if 34 <= n <= 66],
            'high': [n for n in candidates if 67 <= n <= 99]
        }
        
        selected = []
        
        # Take best from each range
        for range_name in ['low', 'medium', 'high']:
            if ranges[range_name]:
                selected.append(ranges[range_name][0])
        
        # Ensure pattern diversity - PHASE 2
        pattern_selected = self.ensure_pattern_diversity(selected, candidates, top_k)
        
        return pattern_selected[:top_k]

    def ensure_pattern_diversity(self, selected, candidates, top_k):
        """Ensure diversity across pattern packs - PHASE 2"""
        if len(selected) >= top_k:
            return selected
        
        # Check for S40 representation - PHASE 2
        s40_candidates = [n for n in candidates if n in self.S40_numbers and n not in selected]
        if s40_candidates and len(selected) < top_k:
            selected.append(s40_candidates[0])
        
        # Check for 3-digit pattern representation - PHASE 2
        for pattern_name, pattern_set in self.pattern_3digit.items():
            pattern_candidates = [n for n in candidates if n in pattern_set and n not in selected]
            if pattern_candidates and len(selected) < top_k:
                selected.append(pattern_candidates[0])
                break
        
        # Check for 4-digit pattern representation - PHASE 2
        for pattern_name, pattern_set in self.pattern_4digit.items():
            pattern_candidates = [n for n in candidates if n in pattern_set and n not in selected]
            if pattern_candidates and len(selected) < top_k:
                selected.append(pattern_candidates[0])
                break
        
        # Check for 6-digit pattern representation (including 164950/164950) - PHASE 2
        for pattern_name, pattern_set in self.pattern_6digit.items():
            pattern_candidates = [n for n in candidates if n in pattern_set and n not in selected]
            if pattern_candidates and len(selected) < top_k:
                selected.append(pattern_candidates[0])
                break
        
        # Fill remaining with highest scores
        remaining = top_k - len(selected)
        if remaining > 0:
            for num in candidates:
                if num not in selected and len(selected) < top_k:
                    selected.append(num)
        
        return selected

    def fallback_prediction(self, numbers, top_k):
        """Fallback prediction"""
        if not numbers:
            return list(range(top_k))
        
        freq = Counter(numbers)
        return [num for num, count in freq.most_common(top_k)]

    # ==================== MAIN PREDICTION METHOD ====================
    def generate_predictions(self, df, days=3, top_k=5):
        """Generate predictions using ultimate ensemble with pattern analysis - PHASE 4"""
        # First, analyze patterns in the data
        self.detect_cross_slot_patterns(df)
        self.analyze_pattern_performance(df)
        self.analyze_cross_script_patterns(df)  # New cross-script analysis
        
        predictions = []
        
        latest_data_date = df['date'].max().date()
        
        # PHASE 4: Determine which slots need prediction for today
        today_predictions = self.get_todays_missing_slots(df, latest_data_date)
        
        print(f"\n🎯 Generating predictions from {latest_data_date}")
        print(f"📅 Today's incomplete slots: {[self.slot_names[s] for s in today_predictions] if today_predictions else 'None'}")
        
        # Predict missing slots for today (PHASE 4)
        for slot in today_predictions:
            slot_name = self.slot_names[slot]
            slot_data = df[df['slot'] == slot]
            numbers = slot_data['number'].tolist()
            
            print(f"   Processing TODAY {slot_name}...")
            
            if len(numbers) < 30:
                pred_numbers = self.fallback_prediction(numbers, top_k)
            else:
                pred_numbers = self.advanced_ensemble(numbers, top_k * 2)
            
            final_pred = self.apply_intelligent_filter(
                {num: 1.0 for num in pred_numbers}, top_k
            )
            
            for rank, number in enumerate(final_pred, 1):
                predictions.append({
                    'date': latest_data_date.strftime('%Y-%m-%d'),
                    'slot': slot_name,
                    'rank': rank,
                    'number': f"{number:02d}",
                    'prediction_type': 'today_missing'
                })
        
        # Predict future days
        start_date = latest_data_date + timedelta(days=1)
        
        for day_offset in range(days):
            target_date = start_date + timedelta(days=day_offset)
            
            for slot in [1, 2, 3, 4]:
                slot_name = self.slot_names[slot]
                slot_data = df[df['slot'] == slot]
                numbers = slot_data['number'].tolist()
                
                print(f"   Processing FUTURE {slot_name} for {target_date}...")
                
                if len(numbers) < 30:
                    pred_numbers = self.fallback_prediction(numbers, top_k)
                else:
                    pred_numbers = self.advanced_ensemble(numbers, top_k * 2)
                
                final_pred = self.apply_intelligent_filter(
                    {num: 1.0 for num in pred_numbers}, top_k
                )
                
                for rank, number in enumerate(final_pred, 1):
                    predictions.append({
                        'date': target_date.strftime('%Y-%m-%d'),
                        'slot': slot_name,
                        'rank': rank,
                        'number': f"{number:02d}",
                        'prediction_type': 'future'
                    })
        
        return pd.DataFrame(predictions)

    def get_todays_missing_slots(self, df, latest_date):
        """PHASE 4: Determine which slots are missing data for the latest date"""
        missing_slots = []

        # Filter to the latest date once to avoid repeated scans
        latest_day_data = df[df['date'].dt.date == latest_date]

        for slot in [1, 2, 3, 4]:
            slot_data = latest_day_data[latest_day_data['slot'] == slot]
            if slot_data.empty:
                missing_slots.append(slot)

        return missing_slots

    def create_output_files(self, predictions_df, df):
        """Create comprehensive output files in organized folders - PHASE 3"""
        # Create wide format
        wide_df = predictions_df.pivot_table(
            index='date',
            columns='slot',
            values='number',
            aggfunc=lambda x: ', '.join(map(str, x))
        ).reset_index()
        
        column_order = ['date'] + [self.slot_names[i] for i in [1, 2, 3, 4]]
        # Ensure all columns exist
        for col in column_order:
            if col not in wide_df.columns:
                wide_df[col] = None
        wide_df = wide_df.reindex(columns=column_order)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to organized folders - PHASE 3
        wide_file = os.path.join(self.output_dir, f'ultimate_predictions_{timestamp}.xlsx')
        detailed_file = os.path.join(self.output_dir, f'ultimate_detailed_{timestamp}.xlsx')
        analysis_file = os.path.join(self.output_dir, f'ultimate_analysis_{timestamp}.txt')
        pattern_file = os.path.join(self.output_dir, f'pattern_analysis_{timestamp}.xlsx')
        cross_script_file = os.path.join(self.output_dir, f'cross_script_analysis_{timestamp}.xlsx')
        ultimate_long_file = os.path.join(
            self.output_dir, f'ultimate_predictions_long_{timestamp}.xlsx'
        )

        wide_df.to_excel(wide_file, index=False)
        predictions_df.to_excel(detailed_file, index=False)
        self._export_ultimate_long(predictions_df, ultimate_long_file)
        
        self.create_analysis_report(predictions_df, df, analysis_file)
        self.save_pattern_analysis(pattern_file)
        self.save_cross_script_analysis(cross_script_file)
        
        # Print actual file paths for user confirmation
        print(f"📁 Organized folder: {self.output_dir}")

        return wide_df

    def _export_ultimate_long(self, predictions_df, filename):
        """Save a long-format ultimate predictions file for interoperability."""
        long_df = predictions_df.copy()
        if 'date' in long_df.columns:
            long_df['DATE'] = pd.to_datetime(long_df['date']).dt.date
        else:
            long_df['DATE'] = None

        long_df['SLOT'] = (
            long_df['slot'].astype(str).str.upper() if 'slot' in long_df else None
        )
        long_df['RANK'] = long_df['rank'] if 'rank' in long_df else None
        long_df['NUMBER'] = (
            long_df['number'].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else None)
            if 'number' in long_df
            else None
        )
        long_df['SOURCE'] = 'SCR6'

        export_cols = ['DATE', 'SLOT', 'RANK', 'NUMBER', 'SOURCE']
        long_df = long_df[export_cols].sort_values(['DATE', 'SLOT', 'RANK'])

        long_df.to_excel(filename, index=False)

    def create_analysis_report(self, predictions_df, df, filename):
        """Create detailed analysis report - PHASE 2 & 3"""
        with open(filename, 'w') as f:
            f.write("ULTIMATE PREDICTION ANALYSIS REPORT - SCR6 ENHANCED\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Data Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n\n")
            
            f.write("ENHANCED PREDICTION METHODOLOGY:\n")
            f.write("- 10-Strategy Ultimate Ensemble with Pattern Integration\n")
            f.write("- Real LSTM with Bidirectional Layers (FIXED retracing)\n")
            f.write("- XGBoost with Advanced Feature Engineering\n")
            f.write("- LightGBM with Gradient Boosting\n")
            f.write("- Neural Network with Deep Learning\n")
            f.write("- Bayesian Probability Analysis\n")
            f.write("- Confidence-Based Gap Analysis\n")
            f.write("- Advanced Pattern Mining\n")
            f.write("- Frequency Momentum Tracking\n")
            f.write("- Multi-step Markov Chains\n")
            f.write("- Statistical Forest Simulation\n\n")
            
            f.write("ADVANCED PATTERN INTEGRATION:\n")
            f.write(f"- 3-digit pattern families: {len(self.pattern_3digit)}\n")
            f.write(f"- 4-digit pattern families: {len(self.pattern_4digit)}\n")
            f.write(f"- 6-digit pattern families: {len(self.pattern_6digit)} (including 164950/164950)\n")
            f.write(f"- S40 special numbers: {len(self.S40_numbers)}\n")
            f.write("- Cross-slot pattern detection\n")
            f.write("- Cross-script pattern learning\n")
            f.write("- Pattern performance tracking\n\n")
            
            f.write("CROSS-SCRIPT INTELLIGENCE:\n")
            f.write("- Tracks when script predictions hit in other slots/dates\n")
            f.write("- Learns script effectiveness patterns\n")
            f.write("- Uses cross-script knowledge for boosting\n\n")
            
            f.write("PARTIAL-DAY PREDICTION SCHEDULE:\n")
            f.write("- Automatically detects incomplete days\n")
            f.write("- Predicts missing slots for current day\n")
            f.write("- Maintains future day predictions\n\n")
            
            f.write("PERFORMANCE OPTIMIZATIONS:\n")
            f.write("- Aggressive TensorFlow warning suppression\n")
            f.write("- Optimized cross-script pattern loading\n")
            f.write("- Efficient error handling in ML strategies\n")
            f.write("- Organized folder structure with absolute paths\n")

    def save_pattern_analysis(self, filename):
        """Save pattern analysis to Excel - PHASE 2"""
        pattern_data = []
        
        # S40 Analysis - PHASE 2
        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            if slot in self.pattern_performance:
                s40_perf = self.pattern_performance[slot].get('S40', 0)
                pattern_data.append({
                    'slot': slot,
                    'pattern_type': 'S40',
                    'performance_percentage': s40_perf,
                    'pattern_size': len(self.S40_numbers)
                })
        
        # 3-digit pattern analysis - PHASE 2
        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            if slot in self.pattern_performance and '3digit' in self.pattern_performance[slot]:
                for pattern_name, performance in self.pattern_performance[slot]['3digit'].items():
                    pattern_data.append({
                        'slot': slot,
                        'pattern_type': f'3digit_{pattern_name}',
                        'performance_percentage': performance,
                        'pattern_size': len(self.pattern_3digit[pattern_name])
                    })
        
        # 6-digit pattern analysis (including 164950/164950) - PHASE 2
        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            if slot in self.pattern_performance and '6digit' in self.pattern_performance[slot]:
                for pattern_name, performance in self.pattern_performance[slot]['6digit'].items():
                    pattern_data.append({
                        'slot': slot,
                        'pattern_type': f'6digit_{pattern_name}',
                        'performance_percentage': performance,
                        'pattern_size': len(self.pattern_6digit[pattern_name])
                    })
        
        if pattern_data:
            pattern_df = pd.DataFrame(pattern_data)
            pattern_df.to_excel(filename, index=False)

    def save_cross_script_analysis(self, filename):
        """Save cross-script pattern analysis"""
        cross_data = []
        
        # Cross-script patterns
        for pattern_key, number_counts in self.cross_script_patterns.items():
            for number, count in number_counts.most_common(10):
                cross_data.append({
                    'pattern_type': 'cross_script',
                    'pattern_key': pattern_key,
                    'number': number,
                    'frequency': count
                })
        
        # Script effectiveness
        for script_name, slot_data in self.script_effectiveness.items():
            for slot_pattern, effectiveness in slot_data.items():
                cross_data.append({
                    'pattern_type': 'script_effectiveness',
                    'script_name': script_name,
                    'slot_pattern': slot_pattern,
                    'effectiveness': effectiveness
                })
        
        if cross_data:
            cross_df = pd.DataFrame(cross_data)
            cross_df.to_excel(filename, index=False)

    def get_performance_stats(self):
        """Get performance statistics"""
        if os.path.exists(self.performance_log):
            perf_df = pd.read_csv(self.performance_log)
            if len(perf_df) > 0:
                total_predictions = len(perf_df)
                total_hits = perf_df['accuracy'].sum()
                overall_accuracy = (total_hits / total_predictions) * 100
                
                print(f"\n📈 PERFORMANCE HISTORY:")
                print(f"   Total Predictions: {total_predictions}")
                print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
                
                return overall_accuracy
        else:
            print("\n📈 PERFORMANCE HISTORY: No performance data yet")
        return 0

def main():
    print("=== ULTIMATE PREDICTOR SCR6 ENHANCED ===")
    print("🎯 10-Strategy Ensemble + Advanced Patterns + Cross-Script Learning")
    
    predictor = UltimatePredictorPro()
    
    # Use absolute path for Excel file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'number prediction learn.xlsx')
    
    # Load updated data
    df = predictor.load_updated_data(file_path)
    
    if df is not None and len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
        signature = predictor.compute_data_signature(df)
        cache_meta = predictor.load_cache_meta()

        # Show data summary
        print(f"\n📊 DATA SUMMARY:")
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            latest_slot_date = slot_data['date'].max().strftime('%Y-%m-%d') if len(slot_data) else 'N/A'
            print(f"   {predictor.slot_names[slot]}: {len(slot_data)} records")

        # Show performance history
        predictor.get_performance_stats()

        predictions = None
        wide_predictions = None

        if predictor.is_cache_hit(signature, cache_meta):
            print(
                f"ℹ️ SCR6 cache hit – reused predictions for data signature last_date={signature.get('last_date')} "
                f"rows={signature.get('rows')}"
            )
            try:
                predictions = pd.read_excel(predictor.cache_detailed_file)
            except Exception:
                predictions = None

        if predictions is None:
            print("\n🎯 Generating ultimate predictions with cross-script intelligence...")
            predictions = predictor.generate_predictions(df, days=3, top_k=5)
            wide_predictions = predictor.create_output_files(predictions, df)
            predictor.persist_cache(signature, predictions, wide_predictions)
        else:
            wide_predictions = predictor.create_output_files(predictions, df)
            print("✅ SCR6 cache reused successfully (no retraining needed).")

        print("✅ Ultimate predictions generated successfully!")
        print("💾 Files saved to organized folders:")
        print(f"   - {predictor.output_dir}/ultimate_predictions_YYYYMMDD_HHMMSS.xlsx")
        print(f"   - {predictor.output_dir}/ultimate_detailed_YYYYMMDD_HHMMSS.xlsx")
        print(f"   - {predictor.output_dir}/ultimate_analysis_YYYYMMDD_HHMMSS.txt")
        print(f"   - {predictor.output_dir}/pattern_analysis_YYYYMMDD_HHMMSS.xlsx")
        print(f"   - {predictor.output_dir}/cross_script_analysis_YYYYMMDD_HHMMSS.xlsx")

        # Display predictions
        if len(wide_predictions) > 0:
            # Show today's predictions first
            today_date = df['date'].max().date().strftime('%Y-%m-%d')
            today_pred = wide_predictions[wide_predictions['date'] == today_date]
            if len(today_pred) > 0:
                print(f"\n🎲 TODAY'S PREDICTIONS FOR {today_date}:")
                today_row = today_pred.iloc[0]
                for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    if slot_name in today_row and pd.notna(today_row[slot_name]):
                        print(f"   {slot_name}: {today_row[slot_name]}")

            # Show tomorrow's predictions
            tomorrow_date = (df['date'].max().date() + timedelta(days=1)).strftime('%Y-%m-%d')
            tomorrow_pred = wide_predictions[wide_predictions['date'] == tomorrow_date]
            if len(tomorrow_pred) > 0:
                print(f"\n🎲 TOMORROW'S PREDICTIONS FOR {tomorrow_date}:")
                tomorrow_row = tomorrow_pred.iloc[0]
                for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    if slot_name in tomorrow_row and pd.notna(tomorrow_row[slot_name]):
                        print(f"   {slot_name}: {tomorrow_row[slot_name]}")
        
        print("\n🔬 ENHANCED METHODOLOGY:")
        print("   • Real LSTM (TensorFlow retracing FIXED) - PHASE 1")
        print("   • XGBoost • LightGBM • Neural Network")
        print("   • Bayesian • Gap Analysis • Pattern Mining • Momentum")
        print("   • Markov Chains • Statistical Forest")
        print("   • 3-digit, 4-digit & 6-digit pattern families (including 164950/164950) - PHASE 2")
        print("   • S40 special number tracking - PHASE 2")
        print("   • Cross-script pattern learning & intelligence")
        print("   • Organized folder structure - PHASE 3")
        print("   • Partial-day prediction scheduling - PHASE 4")
        print("   • Aggressive TensorFlow suppression - OPTIMIZED")
        print("   • Performance-optimized cross-script loading - OPTIMIZED")
        
    else:
        print("❌ Failed to load data.")


def build_ml_feature_dataframe(history_numbers):
    """Helper to expose SCR6 feature builder for external utilities."""
    predictor = UltimatePredictorPro()
    features, targets = predictor.create_advanced_features(history_numbers)
    if features is None or len(features) == 0:
        return pd.DataFrame()

    columns = [f"feature_{i+1}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=columns)
    df["target"] = targets
    return df


if __name__ == "__main__":
    main()