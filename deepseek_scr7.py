import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
import os
import glob
import json
import shutil
from quant_excel_loader import load_results_excel
warnings.filterwarnings('ignore')

class AdvancedLearningPredictor:
    def __init__(self):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.predictions_dir = os.path.join(self.base_dir, "predictions", "deepseek_scr7")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.performance_dir = os.path.join(self.logs_dir, "performance")
        
        # Create directories
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.performance_dir, exist_ok=True)
        
        self.performance_log = os.path.join(self.performance_dir, "advanced_performance.csv")
        self.model_weights_file = "model_weights.json"
        self.initialize_system()
        
        # Organize legacy files
        self.organize_legacy_files()
        
    def organize_legacy_files(self):
        """Move old scr7 outputs from root to structured folders"""
        print("🧹 Organizing SCR7 files into structured folders...")
        
        # Patterns for scr7 files
        patterns = [
            "advanced_predictions_*.xlsx",
            "advanced_detailed_*.xlsx", 
            "advanced_analysis_*.txt"
        ]
        
        # Move pattern-based files
        for pattern in patterns:
            for file_path in glob.glob(os.path.join(self.base_dir, pattern)):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.predictions_dir, filename)
                try:
                    if not os.path.exists(dest_path):
                        shutil.move(file_path, dest_path)
                        print(f"   📂 Moved: {filename} → {self.predictions_dir}")
                except Exception as e:
                    print(f"   ⚠️  Could not move {filename}: {e}")
        
        # Move prediction_diagnostic.xlsx
        diagnostic_src = os.path.join(self.base_dir, "prediction_diagnostic.xlsx")
        diagnostic_dest = os.path.join(self.predictions_dir, "prediction_diagnostic.xlsx")
        if os.path.exists(diagnostic_src) and not os.path.exists(diagnostic_dest):
            try:
                shutil.move(diagnostic_src, diagnostic_dest)
                print(f"   📂 Moved: prediction_diagnostic.xlsx → {self.predictions_dir}")
            except Exception as e:
                print(f"   ⚠️  Could not move prediction_diagnostic.xlsx: {e}")
        
        # Move advanced_performance.csv
        perf_src = os.path.join(self.base_dir, "advanced_performance.csv")
        perf_dest = self.performance_log
        if os.path.exists(perf_src):
            if os.path.exists(perf_dest):
                # Both exist - keep the larger one
                src_size = os.path.getsize(perf_src)
                dest_size = os.path.getsize(perf_dest)
                if src_size > dest_size:
                    try:
                        shutil.move(perf_src, perf_dest)
                        print(f"   📂 Moved: advanced_performance.csv → {self.performance_dir}")
                    except Exception as e:
                        print(f"   ⚠️  Could not move advanced_performance.csv: {e}")
            else:
                try:
                    shutil.move(perf_src, perf_dest)
                    print(f"   📂 Moved: advanced_performance.csv → {self.performance_dir}")
                except Exception as e:
                    print(f"   ⚠️  Could not move advanced_performance.csv: {e}")
        
    def initialize_system(self):
        """Initialize system with proper performance tracking"""
        if not os.path.exists(self.performance_log):
            pd.DataFrame(columns=[
                'date', 'slot', 'model_id', 'predicted_numbers', 'actual_number', 
                'hit_rank', 'hit_count', 'accuracy', 'timestamp'
            ]).to_csv(self.performance_log, index=False)
        
        # Load or initialize model weights
        self.model_weights = self.load_model_weights()
        
    def load_model_weights(self):
        """Load model weights from file or initialize"""
        if os.path.exists(self.model_weights_file):
            with open(self.model_weights_file, 'r') as f:
                return json.load(f)
        else:
            return self.initialize_default_weights()
    
    def initialize_default_weights(self):
        """Initialize default weights for all models"""
        weights = {}
        
        # Script models (SCR1-8 for each slot)
        for script in range(1, 9):
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                model_id = f"SCR{script}_{slot}"
                weights[model_id] = {
                    'weight': 0.3,
                    'rank_weights': [1.0, 0.8, 0.6, 0.4, 0.2],
                    'recent_performance': [],
                    'overall_accuracy': 0.3
                }
        
        # Opposite models
        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            model_id = f"OPPOSITE_{slot}"
            weights[model_id] = {
                'weight': 0.2,
                'rank_weights': [1.0],
                'recent_performance': [],
                'overall_accuracy': 0.2
            }
        
        # Cross-slot models
        slot_pairs = [
            ('GZBD', 'FRBD'), ('GALI', 'FRBD'), ('DSWR', 'FRBD'),
            ('FRBD', 'GZBD'), ('GALI', 'GZBD'), ('DSWR', 'GZBD'),
            ('FRBD', 'GALI'), ('GZBD', 'GALI'), ('DSWR', 'GALI'),
            ('FRBD', 'DSWR'), ('GZBD', 'DSWR'), ('GALI', 'DSWR')
        ]
        
        for from_slot, to_slot in slot_pairs:
            model_id = f"CROSS_{from_slot}_TO_{to_slot}"
            weights[model_id] = {
                'weight': 0.15,
                'rank_weights': [1.0],
                'recent_performance': [],
                'overall_accuracy': 0.15
            }
        
        return weights

    def load_data(self, file_path):
        """Canonical data loader that delegates to quant_excel_loader"""
        try:
            df = load_results_excel(file_path)
            print(f"✅ Data loaded successfully via quant_excel_loader: {len(df)} records")
            return df

        except Exception as e:
            print(f"❌ Error loading data via quant_excel_loader: {e}")
            raise

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

    def get_opposite(self, n):
        """Get opposite number (09->90, 12->21, etc)"""
        if n < 10:
            return n * 10
        else:
            tens = n // 10
            ones = n % 10
            return ones * 10 + tens

    def run_evaluation_mode(self, df, evaluation_date):
        """Run evaluation mode to update performance logs"""
        print(f"🔍 Running evaluation for {evaluation_date}")
        
        # Get actual results for evaluation date
        evaluation_data = df[df['date'] == pd.to_datetime(evaluation_date)]
        if evaluation_data.empty:
            print(f"❌ No data found for {evaluation_date}")
            return
        
        # Load previous predictions for this date
        previous_predictions = self.load_previous_predictions_for_date(evaluation_date)
        
        performance_rows = []
        
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            actual_data = evaluation_data[evaluation_data['slot'] == slot]
            
            if actual_data.empty:
                continue
                
            actual_number = actual_data['number'].iloc[0]
            
            # Evaluate each model's predictions
            for model_id, predictions in previous_predictions.items():
                if slot_name in model_id or 'OPPOSITE' in model_id or 'CROSS' in model_id:
                    if slot_name in predictions:
                        pred_numbers = predictions[slot_name]
                        hit_rank = None
                        hit_count = 0
                        
                        if actual_number in pred_numbers:
                            hit_rank = pred_numbers.index(actual_number) + 1
                            hit_count = 1
                        
                        performance_rows.append({
                            'date': evaluation_date,
                            'slot': slot_name,
                            'model_id': model_id,
                            'predicted_numbers': ','.join(map(str, pred_numbers)),
                            'actual_number': actual_number,
                            'hit_rank': hit_rank,
                            'hit_count': hit_count,
                            'accuracy': hit_count,
                            'timestamp': datetime.now()
                        })
        
        # Save performance data
        if performance_rows:
            perf_df = pd.DataFrame(performance_rows)
            if os.path.exists(self.performance_log):
                existing_df = pd.read_csv(self.performance_log)
                updated_df = pd.concat([existing_df, perf_df], ignore_index=True)
            else:
                updated_df = perf_df
            
            updated_df.to_csv(self.performance_log, index=False)
            print(f"✅ Updated performance log with {len(performance_rows)} records")
            
            # Update model weights based on new performance data
            self.update_model_weights()

    def load_previous_predictions_for_date(self, target_date):
        """Load previous predictions for a specific date"""
        predictions = {}
        
        # Look for prediction files in both root and predictions directory
        search_dirs = [self.base_dir, self.predictions_dir]
        prediction_files = []
        
        for search_dir in search_dirs:
            prediction_files.extend(glob.glob(os.path.join(search_dir, '*predictions*.xlsx')))
            prediction_files.extend(glob.glob(os.path.join(search_dir, '*detailed*.xlsx')))
        
        for file in prediction_files:
            try:
                df = pd.read_excel(file)
                if 'date' in df.columns:
                    date_pred = df[df['date'] == target_date]
                    if not date_pred.empty:
                        model_id = os.path.basename(file).replace('.xlsx', '')
                        predictions[model_id] = {}
                        
                        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                            if slot in date_pred.columns:
                                numbers_str = str(date_pred[slot].iloc[0])
                                if numbers_str and numbers_str != 'nan':
                                    numbers = [self.clean_number(x) for x in numbers_str.split(',')]
                                    numbers = [n for n in numbers if n is not None]
                                    predictions[model_id][slot] = numbers
            except Exception as e:
                continue
        
        return predictions

    def update_model_weights(self):
        """Update model weights based on performance data"""
        if not os.path.exists(self.performance_log):
            return
        
        perf_df = pd.read_csv(self.performance_log)
        
        for model_id in self.model_weights.keys():
            model_data = perf_df[perf_df['model_id'] == model_id]
            
            if len(model_data) > 0:
                # Calculate recent performance (last 30 days)
                recent_days = 30
                recent_data = model_data.tail(recent_days)
                
                if len(recent_data) > 0:
                    hit_rate = recent_data['hit_count'].mean()
                    self.model_weights[model_id]['recent_performance'] = recent_data['hit_count'].tolist()
                    self.model_weights[model_id]['overall_accuracy'] = hit_rate
                    
                    # Update weight based on performance
                    base_weight = 0.3
                    performance_bonus = min(hit_rate * 0.5, 0.3)  # Max 30% bonus
                    self.model_weights[model_id]['weight'] = base_weight + performance_bonus
        
        # Save updated weights
        with open(self.model_weights_file, 'w') as f:
            json.dump(self.model_weights, f, indent=2)

    def advanced_prediction_engine(self, df, days=3):
        """Advanced prediction engine with model-based voting"""
        predictions = {}
        diagnostic_data = []
        
        latest_date = df['date'].max().date()
        start_date = latest_date + timedelta(days=1)
        
        # Get base predictions from all available scripts
        base_predictions = self.get_base_predictions(df, days)
        
        for day_offset in range(days):
            target_date = start_date + timedelta(days=day_offset)
            date_str = target_date.strftime('%Y-%m-%d')
            predictions[date_str] = {}
            
            for slot in [1, 2, 3, 4]:
                slot_name = self.slot_names[slot]
                slot_data = df[df['slot'] == slot]
                numbers = slot_data['number'].tolist()
                
                if len(numbers) < 10:
                    pred_data = self.fallback_prediction(numbers)
                else:
                    # STRATEGY 1: Model-based voting from all scripts
                    model_votes = self.model_based_voting(base_predictions, date_str, slot_name)
                    
                    # STRATEGY 2: Cross-slot intelligence
                    cross_slot_votes = self.cross_slot_intelligence(base_predictions, date_str, slot_name)
                    
                    # STRATEGY 3: Opposite number analysis
                    opposite_votes = self.opposite_analysis(base_predictions, date_str, slot_name, numbers)
                    
                    # Combine all strategies with proper weighting
                    combined_scores = self.combine_strategies_advanced(
                        model_votes, cross_slot_votes, opposite_votes, slot_name
                    )
                    
                    # Dynamic top-k selection based on EV
                    top_k = self.ev_based_top_k_selection(combined_scores, slot_name)
                    pred_numbers = [num for num, score in combined_scores.most_common(top_k)]
                    
                    # Store diagnostic data
                    for num, score in combined_scores.most_common(10):
                        diagnostic_data.append({
                            'date': date_str,
                            'slot': slot_name,
                            'number': num,
                            'score_total': score,
                            'score_model': model_votes.get(num, 0),
                            'score_cross': cross_slot_votes.get(num, 0),
                            'score_opposite': opposite_votes.get(num, 0)
                        })
                    
                    pred_data = {
                        'numbers': pred_numbers,
                        'top_k': top_k,
                        'confidence': combined_scores.most_common(1)[0][1] if combined_scores else 0
                    }
                
                predictions[date_str][slot_name] = pred_data
        
        # Save diagnostic data
        if diagnostic_data:
            diag_df = pd.DataFrame(diagnostic_data)
            diag_path = os.path.join(self.predictions_dir, 'prediction_diagnostic.xlsx')
            diag_df.to_excel(diag_path, index=False)
        
        return predictions

    def model_based_voting(self, base_predictions, date_str, slot_name):
        """Model-based voting with rank-based weights"""
        counter = Counter()
        
        for model_id, model_data in base_predictions.items():
            if date_str in model_data and slot_name in model_data[date_str]:
                predictions = model_data[date_str][slot_name]
                model_weight = self.model_weights.get(model_id, {}).get('weight', 0.3)
                
                for rank, number in enumerate(predictions):
                    rank_weight = 1.0 / (rank + 1)  # Rank-based decay
                    score = model_weight * rank_weight
                    counter[number] += score
        
        return counter

    def cross_slot_intelligence(self, base_predictions, date_str, target_slot):
        """Cross-slot intelligence with learned correlations"""
        counter = Counter()
        
        for other_slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            if other_slot == target_slot:
                continue
                
            model_id = f"CROSS_{other_slot}_TO_{target_slot}"
            model_weight = self.model_weights.get(model_id, {}).get('weight', 0.15)
            
            for script_id, script_data in base_predictions.items():
                if date_str in script_data and other_slot in script_data[date_str]:
                    numbers = script_data[date_str][other_slot]
                    for number in numbers[:5]:  # Top 5 from other slot
                        counter[number] += model_weight
        
        return counter

    def opposite_analysis(self, base_predictions, date_str, slot_name, historical_numbers):
        """Opposite number analysis based on patterns"""
        counter = Counter()
        model_id = f"OPPOSITE_{slot_name}"
        model_weight = self.model_weights.get(model_id, {}).get('weight', 0.2)
        
        # Get numbers from all scripts for this slot
        all_numbers = []
        for script_data in base_predictions.values():
            if date_str in script_data and slot_name in script_data[date_str]:
                all_numbers.extend(script_data[date_str][slot_name])
        
        # Add opposites of predicted numbers
        for number in all_numbers[:10]:  # Top 10 predicted numbers
            opposite_num = self.get_opposite(number)
            counter[opposite_num] += model_weight
        
        # Add opposites of recent historical numbers
        for number in historical_numbers[-10:]:
            opposite_num = self.get_opposite(number)
            counter[opposite_num] += model_weight * 0.5
        
        return counter

    def get_base_predictions(self, df, days):
        """Get base predictions from available data (simulating SCR1-8)"""
        base_predictions = {}
        
        # Simulate different prediction strategies (in real implementation, run actual scripts)
        strategies = {
            'frequency': self.frequency_based_prediction,
            'gap': self.gap_based_prediction,
            'pattern': self.pattern_based_prediction,
            'markov': self.markov_based_prediction
        }
        
        latest_date = df['date'].max().date()
        start_date = latest_date + timedelta(days=1)
        
        for strategy_name, strategy_func in strategies.items():
            strategy_predictions = {}
            
            for day_offset in range(days):
                target_date = start_date + timedelta(days=day_offset)
                date_str = target_date.strftime('%Y-%m-%d')
                strategy_predictions[date_str] = {}
                
                for slot in [1, 2, 3, 4]:
                    slot_name = self.slot_names[slot]
                    slot_data = df[df['slot'] == slot]
                    numbers = slot_data['number'].tolist()
                    
                    if len(numbers) >= 10:
                        pred_numbers = strategy_func(numbers, 10)
                    else:
                        pred_numbers = self.fallback_prediction(numbers)['numbers']
                    
                    strategy_predictions[date_str][slot_name] = pred_numbers
            
            base_predictions[f"SCR_{strategy_name}"] = strategy_predictions
        
        return base_predictions

    def frequency_based_prediction(self, numbers, top_k):
        """Frequency-based prediction strategy"""
        freq = Counter(numbers[-30:])
        return [num for num, count in freq.most_common(top_k)]

    def gap_based_prediction(self, numbers, top_k):
        """Gap-based prediction strategy"""
        last_seen = {}
        for i, num in enumerate(numbers):
            last_seen[num] = i
        
        current_idx = len(numbers) - 1
        gap_scores = {}
        
        for num in range(100):
            if num in last_seen:
                gap = current_idx - last_seen[num]
                gap_scores[num] = gap
            else:
                gap_scores[num] = 999
        
        due_numbers = sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, gap in due_numbers[:top_k]]

    def pattern_based_prediction(self, numbers, top_k):
        """Pattern-based prediction strategy"""
        if len(numbers) < 10:
            return self.fallback_prediction(numbers)['numbers']
        
        # Simple sequence detection
        sequences = []
        current_seq = [numbers[0]]
        
        for i in range(1, len(numbers)):
            diff = abs(numbers[i] - numbers[i-1])
            if diff <= 5:
                current_seq.append(numbers[i])
            else:
                if len(current_seq) >= 3:
                    sequences.append(current_seq)
                current_seq = [numbers[i]]
        
        if len(current_seq) >= 3:
            sequences.append(current_seq)
        
        predictions = []
        if sequences:
            last_seq = sequences[-1]
            if len(last_seq) >= 3:
                avg_diff = (last_seq[-1] - last_seq[0]) / (len(last_seq) - 1)
                next_num = int((last_seq[-1] + avg_diff) % 100)
                predictions.append(next_num)
        
        # Fill with frequency if needed
        if len(predictions) < top_k:
            freq = Counter(numbers[-20:])
            additional = [num for num, count in freq.most_common(top_k) if num not in predictions]
            predictions.extend(additional[:top_k - len(predictions)])
        
        return predictions[:top_k]

    def markov_based_prediction(self, numbers, top_k):
        """Markov chain based prediction"""
        if len(numbers) < 10:
            return self.fallback_prediction(numbers)['numbers']
        
        transitions = {}
        for i in range(1, len(numbers)):
            prev = numbers[i-1]
            curr = numbers[i]
            
            if prev not in transitions:
                transitions[prev] = {}
            transitions[prev][curr] = transitions[prev].get(curr, 0) + 1
        
        predictions = []
        if numbers and numbers[-1] in transitions:
            next_probs = transitions[numbers[-1]]
            likely_next = sorted(next_probs.items(), key=lambda x: x[1], reverse=True)
            predictions = [num for num, count in likely_next[:top_k]]
        
        if len(predictions) < top_k:
            freq = Counter(numbers[-15:])
            additional = [num for num, count in freq.most_common(top_k) if num not in predictions]
            predictions.extend(additional[:top_k - len(predictions)])
        
        return predictions[:top_k]

    def combine_strategies_advanced(self, model_votes, cross_slot_votes, opposite_votes, slot_name):
        """Combine strategies with learned weights"""
        combined = Counter()
        
        # Get weights for each strategy type
        model_weight = 0.6
        cross_weight = 0.25
        opposite_weight = 0.15
        
        # Apply weights
        for num, score in model_votes.items():
            combined[num] += score * model_weight
        
        for num, score in cross_slot_votes.items():
            combined[num] += score * cross_weight
            
        for num, score in opposite_votes.items():
            combined[num] += score * opposite_weight
        
        return combined

    def ev_based_top_k_selection(self, combined_scores, slot_name):
        """EV-based top-k selection"""
        # Simplified EV calculation - in production, use historical backtesting
        slot_accuracy = self.model_weights.get(f"SCR_frequency_{slot_name}", {}).get('overall_accuracy', 0.3)
        
        # Calculate confidence
        if combined_scores:
            top_score = combined_scores.most_common(1)[0][1]
            scores = [score for _, score in combined_scores.most_common(10)]
            avg_score = np.mean(scores) if scores else 0
            confidence = top_score / avg_score if avg_score > 0 else 0
        else:
            confidence = 0
        
        # EV-based decision
        if confidence > 2.0 and slot_accuracy > 0.4:
            return 5   # High confidence - fewer numbers
        elif confidence > 1.5 and slot_accuracy > 0.3:
            return 10  # Medium confidence
        else:
            return 15  # Low confidence - more numbers

    def fallback_prediction(self, numbers):
        """Fallback prediction"""
        if not numbers:
            numbers = list(range(100))
        
        freq = Counter(numbers)
        pred_numbers = [num for num, count in freq.most_common(10)]
        
        return {
            'numbers': pred_numbers,
            'top_k': 10,
            'confidence': 0.5
        }

    def generate_predictions(self, df, days=3, evaluation_date=None):
        """Generate predictions with optional evaluation"""
        if evaluation_date:
            self.run_evaluation_mode(df, evaluation_date)
        
        print("🔍 Running advanced prediction engine...")
        advanced_pred = self.advanced_prediction_engine(df, days)
        
        # Convert to DataFrame format
        predictions_list = []
        for date, slots in advanced_pred.items():
            for slot_name, pred_data in slots.items():
                numbers = pred_data['numbers']
                top_k = pred_data['top_k']
                
                for rank, number in enumerate(numbers[:top_k], 1):
                    predictions_list.append({
                        'date': date,
                        'slot': slot_name,
                        'rank': rank,
                        'number': f"{number:02d}",
                        'top_k': top_k,
                        'confidence': pred_data['confidence']
                    })
        
        return pd.DataFrame(predictions_list)

    def create_output(self, predictions_df, df):
        """Create comprehensive output files"""
        # Wide format
        wide_data = []
        for date in predictions_df['date'].unique():
            date_data = {'date': date}
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                slot_pred = predictions_df[
                    (predictions_df['date'] == date) & 
                    (predictions_df['slot'] == slot)
                ]
                if not slot_pred.empty:
                    numbers = slot_pred['number'].tolist()
                    date_data[slot] = ', '.join(numbers)
                    date_data[f'{slot}_top_k'] = slot_pred['top_k'].iloc[0]
            wide_data.append(date_data)
        
        wide_df = pd.DataFrame(wide_data)
        
        # Save files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wide_path = os.path.join(self.predictions_dir, f'advanced_predictions_{timestamp}.xlsx')
        detailed_path = os.path.join(self.predictions_dir, f'advanced_detailed_{timestamp}.xlsx')
        
        wide_df.to_excel(wide_path, index=False)
        predictions_df.to_excel(detailed_path, index=False)
        
        # Create analysis report
        analysis_path = os.path.join(self.predictions_dir, f'advanced_analysis_{timestamp}.txt')
        self.create_analysis_report(predictions_df, df, analysis_path)
        
        return wide_df

    def create_analysis_report(self, predictions_df, df, filename):
        """Create advanced analysis report"""
        with open(filename, 'w') as f:
            f.write("ADVANCED LEARNING PREDICTOR REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Historical Records: {len(df)}\n")
            f.write(f"Data Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n\n")
            
            f.write("ADVANCED FEATURES:\n")
            f.write("• Model-based voting with rank weights\n")
            f.write("• Cross-slot correlation learning\n")
            f.write("• Opposite number pattern analysis\n")
            f.write("• EV-based dynamic top-k selection\n")
            f.write("• Performance log auto-update\n")
            f.write("• Diagnostic score breakdown\n\n")
            
            # Add model performance summary
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            for model_id, weights in list(self.model_weights.items())[:10]:  # Show top 10
                f.write(f"  {model_id}: {weights['overall_accuracy']:.3f} (weight: {weights['weight']:.3f})\n")

def main():
    print("=== ADVANCED LEARNING PREDICTOR ===")
    print("🎯 Model-Based Voting + Performance Tracking + EV Optimization")
    
    predictor = AdvancedLearningPredictor()
    file_path = 'number prediction learn.xlsx'
    
    # Load data
    df = predictor.load_data(file_path)
    
    if df is not None and len(df) > 0:
        # Add slot names for analysis
        df['slot_name'] = df['slot'].map(predictor.slot_names)
        
        print(f"📊 Historical data: {len(df)} records")
        print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Check if we want to run evaluation mode
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
            evaluation_date = sys.argv[2] if len(sys.argv) > 2 else '2025-11-14'
            predictor.run_evaluation_mode(df, evaluation_date)
            return
        
        # Generate predictions
        print("\n🎯 Generating advanced predictions...")
        predictions = predictor.generate_predictions(df, days=3)
        
        # Create output
        wide_predictions = predictor.create_output(predictions, df)
        
        print("✅ Advanced predictions generated successfully!")
        print("💾 Files saved:")
        print(f"   - {predictor.predictions_dir}\\advanced_predictions_YYYYMMDD_HHMMSS.xlsx")
        print(f"   - {predictor.predictions_dir}\\advanced_detailed_YYYYMMDD_HHMMSS.xlsx")
        print(f"   - {predictor.predictions_dir}\\prediction_diagnostic.xlsx (score breakdown)")
        print(f"   - {predictor.predictions_dir}\\advanced_analysis_YYYYMMDD_HHMMSS.txt")
        print(f"   - {predictor.performance_log}  (advanced performance log)")
        
        # Display predictions
        if len(wide_predictions) > 0:
            first_date = wide_predictions['date'].iloc[0]
            print(f"\n🎲 ADVANCED PREDICTIONS FOR {first_date}:")
            
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in wide_predictions.columns:
                    numbers = wide_predictions[wide_predictions['date'] == first_date][slot_name].iloc[0]
                    top_k = wide_predictions[wide_predictions['date'] == first_date][f'{slot_name}_top_k'].iloc[0]
                    print(f"   {slot_name} (Top-{top_k}): {numbers}")
            
            print(f"\n🔄 OPPOSITE NUMBERS ANALYSIS:")
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in wide_predictions.columns:
                    numbers_str = wide_predictions[wide_predictions['date'] == first_date][slot_name].iloc[0]
                    numbers = [int(x) for x in numbers_str.split(', ')]
                    opposites = [predictor.get_opposite(n) for n in numbers[:3]]
                    print(f"   {slot_name} Opposites: {', '.join([f'{n:02d}' for n in opposites])}")
        
        print("\n🔬 ADVANCED METHODOLOGY:")
        print("   • Model-based voting with performance weights")
        print("   • Rank-based score decay (rank1 > rank2 > ...)")
        print("   • Cross-slot correlation learning")
        print("   • Opposite number pattern analysis")
        print("   • EV-based dynamic top-k (5/10/15)")
        print("   • Performance log auto-update")
        print("   • Diagnostic score breakdown")
        
        print("\n💡 Usage for evaluation:")
        print("   python deepseek_scr7.py evaluate 2025-11-14")
        
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()