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

def organize_old_scr10_files():
    """Move old SCR10 files from root to structured folders"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, "predictions", "deepseek_scr8")
    logs_dir = os.path.join(base_dir, "logs", "performance")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue

        lower = fname.lower()

        # Move scr10 prediction / detailed / analysis / diagnostic files
        if lower.startswith("scr10_predictions_") and lower.endswith(".xlsx"):
            shutil.move(fpath, os.path.join(pred_dir, fname))
        elif lower.startswith("scr10_detailed_") and lower.endswith(".xlsx"):
            shutil.move(fpath, os.path.join(pred_dir, fname))
        elif lower.startswith("scr10_analysis_") and lower.endswith(".txt"):
            shutil.move(fpath, os.path.join(pred_dir, fname))
        elif lower == "scr10_diagnostic.xlsx":
            shutil.move(fpath, os.path.join(pred_dir, fname))
        # Move performance log
        elif lower == "scr10_performance.csv":
            shutil.move(fpath, os.path.join(logs_dir, fname))

class SCR10UltimatePredictor:
    def __init__(self):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.predictions_dir = os.path.join(self.base_dir, "predictions", "deepseek_scr8")
        self.logs_dir = os.path.join(self.base_dir, "logs", "performance")
        
        # Create directories
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.performance_log = os.path.join(self.logs_dir, "scr10_performance.csv")
        self.model_weights_file = "scr10_weights.json"
        
        # Pattern packs from analysis
        self.SLOT_HOT = {
            "FRBD": [1, 3, 26, 44, 64, 71, 77, 82, 84],
            "GZBD": [20, 23, 50, 52, 67, 89],
            "GALI": [23, 28, 31, 32, 42, 57, 64, 72, 80, 94, 95],
            "DSWR": [25, 36, 48, 55, 68, 70, 88, 94, 96],
        }
        
        self.SLOT_COLD = {
            "FRBD": [21, 25, 48, 53, 61, 65, 85, 86, 87, 93, 97],
            "GZBD": [0, 1, 5, 12, 18, 21, 32, 46, 47, 48, 55, 66, 68, 84],
            "GALI": [0, 4, 10, 18, 29, 33, 37, 38, 43, 44, 48, 51, 52, 61, 65, 76, 77, 78, 88, 89],
            "DSWR": [7, 13, 19, 23, 27, 28, 32, 39, 52, 65, 69, 72, 81, 86, 95, 97],
        }
        
        self.DIGIT_BIAS = {
            "FRBD": {
                "tens": [0, 7, 4, 6, 5],
                "ones": [4, 2, 8, 0, 1],
                "sums": [8, 10, 9, 6, 12],
            },
            "GZBD": {
                "tens": [9, 5, 2, 6, 8],
                "ones": [3, 9, 0, 1, 4],
                "sums": [9, 8, 6, 7, 11],
            },
            "GALI": {
                "tens": [6, 9, 4, 3, 2],
                "ones": [2, 4, 1, 5, 7],
                "sums": [10, 9, 5, 8, 12],
            },
            "DSWR": {
                "tens": [9, 5, 7, 8, 0],
                "ones": [8, 6, 4, 3, 1],
                "sums": [9, 7, 10, 8, 6],
            },
        }
        
        self.GLOBAL_MULTI_SLOT_HOT = [
            71, 96, 64, 94, 3, 57, 67, 68, 26, 70,
            72, 80, 84, 28, 37, 42, 50, 63, 88, 98,
        ]
        
        self.RANGE_STATS = {
            "FRBD": {"low_pct": 0.35, "mid_pct": 0.33, "high_pct": 0.32},
            "GZBD": {"low_pct": 0.31, "mid_pct": 0.34, "high_pct": 0.35},
            "GALI": {"low_pct": 0.33, "mid_pct": 0.32, "high_pct": 0.35},
            "DSWR": {"low_pct": 0.29, "mid_pct": 0.34, "high_pct": 0.37},
        }
        
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize system with pattern-based learning"""
        if not os.path.exists(self.performance_log):
            pd.DataFrame(columns=[
                'date', 'slot', 'model_id', 'predicted_numbers', 'actual_number', 
                'hit_rank', 'hit_count', 'accuracy', 'timestamp'
            ]).to_csv(self.performance_log, index=False)
        
        self.model_weights = self.load_model_weights()
    
    def load_model_weights(self):
        """Load or initialize model weights"""
        if os.path.exists(self.model_weights_file):
            with open(self.model_weights_file, 'r') as f:
                return json.load(f)
        else:
            return self.initialize_pattern_weights()
    
    def initialize_pattern_weights(self):
        """Initialize weights based on pattern analysis"""
        weights = {}
        
        # Pattern-based models
        pattern_models = {
            'hot_cold': 0.25,
            'digit_bias': 0.20,
            'global_hot': 0.15,
            'cross_slot': 0.20,
            'opposite': 0.20
        }
        
        for model, weight in pattern_models.items():
            weights[model] = {
                'weight': weight,
                'overall_accuracy': 0.3,
                'recent_performance': []
            }
        
        return weights

    def load_data(self, file_path):
        """Load data via canonical loader and auto-update pattern packs."""
        try:
            df = load_results_excel(file_path)
            if df is None or len(df) == 0:
                print("❌ SCR8: No valid data from quant_excel_loader")
                return None
            print(f"✅ SCR8: Loaded {len(df)} records via quant_excel_loader")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def update_pattern_packs(self, df):
        """Auto-update pattern packs with latest data"""
        # This can be expanded to automatically recalculate HOT/COLD patterns
        print("🔄 Pattern packs updated with latest data")
        # For now using static patterns, but can be made dynamic

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

    def pattern_based_scoring(self, slot_name, numbers, base_models_score):
        """Advanced pattern-based scoring"""
        scores = Counter()
        
        for num in range(100):
            score = 0.0
            
            # 1. Base ensemble models
            score += base_models_score.get(num, 0.0)
            
            # 2. Slot-specific HOT/COLD patterns
            if num in self.SLOT_HOT[slot_name]:
                score += 2.0
            if num in self.SLOT_COLD[slot_name]:
                score -= 1.0
            
            # 3. Global multi-slot HOT
            if num in self.GLOBAL_MULTI_SLOT_HOT:
                score += 1.0
            
            # 4. Digit pattern bias
            score += self.digit_bonus(slot_name, num)
            
            # 5. Cross-slot + opposite bonus
            score += self.cross_slot_opp_bonus(slot_name, num)
            
            if score > 0:
                scores[num] = score
        
        return scores

    def digit_bonus(self, slot_name, num):
        """Digit pattern bonus calculation"""
        tens = num // 10
        ones = num % 10
        digit_sum = tens + ones
        
        bias = self.DIGIT_BIAS[slot_name]
        bonus = 0.0
        
        if tens in bias["tens"]:
            bonus += 0.5
        if ones in bias["ones"]:
            bonus += 0.5
        if digit_sum in bias["sums"]:
            bonus += 0.5
            
        return bonus

    def cross_slot_opp_bonus(self, slot_name, num):
        """Cross-slot and opposite bonus"""
        bonus = 0.0
        opposite_num = self.get_opposite(num)
        
        # Check if opposite appears in other slots' HOT lists
        for other_slot, hot_list in self.SLOT_HOT.items():
            if other_slot == slot_name:
                continue
            if opposite_num in hot_list:
                bonus += 0.7
        
        return bonus

    def enforce_range_diversity(self, candidates, top_k):
        """Enforce range diversity in final selection"""
        if len(candidates) <= top_k:
            return candidates[:top_k]
        
        lows = [n for n in candidates if 0 <= n <= 33]
        mids = [n for n in candidates if 34 <= n <= 66]
        highs = [n for n in candidates if 67 <= n <= 99]
        
        selected = []
        
        # Take at least one from each range
        for group in [lows, mids, highs]:
            if group and len(selected) < top_k:
                selected.append(group[0])
        
        # Fill remaining with highest scores
        remaining = top_k - len(selected)
        if remaining > 0:
            for num in candidates:
                if num not in selected and len(selected) < top_k:
                    selected.append(num)
        
        return selected[:top_k]

    def fetch_online_predictions(self):
        """Fetch predictions from online sources (placeholder)"""
        online_predictions = {}
        
        try:
            # Example: You can integrate with Telegram bots or websites here
            # Using requests with your API keys
            
            # Placeholder structure - implement based on your online sources
            online_sources = {
                # "source1": "https://api.example.com/predictions",
                # "source2": "https://bot.telegram.org/data"
            }
            
            for source_name, url in online_sources.items():
                try:
                    # response = requests.get(url, headers=your_headers)
                    # data = response.json()
                    # Process the data and add to online_predictions
                    pass
                except Exception as e:
                    print(f"⚠️ Failed to fetch from {source_name}: {e}")
            
        except Exception as e:
            print(f"⚠️ Online predictions skipped: {e}")
        
        return online_predictions

    def advanced_prediction_engine(self, df, days=3):
        """Ultimate prediction engine with all patterns"""
        predictions = {}
        diagnostic_data = []
        
        latest_date = df['date'].max().date()
        start_date = latest_date + timedelta(days=1)
        
        # Get online predictions if available
        online_predictions = self.fetch_online_predictions()
        
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
                    # Get base predictions from multiple strategies
                    base_strategies = self.get_base_strategies(numbers)
                    
                    # Apply pattern-based scoring
                    pattern_scores = self.pattern_based_scoring(
                        slot_name, numbers, base_strategies
                    )
                    
                    # Add online predictions if available
                    if online_predictions:
                        online_scores = self.integrate_online_predictions(
                            online_predictions, slot_name
                        )
                        for num, score in online_scores.items():
                            pattern_scores[num] += score * 0.3  # 30% weight for online
                    
                    # Dynamic top-k selection
                    top_k = self.smart_top_k_selection(pattern_scores, slot_name)
                    pred_numbers = [num for num, score in pattern_scores.most_common(top_k * 2)]
                    
                    # Apply range diversity
                    final_pred = self.enforce_range_diversity(pred_numbers, top_k)
                    
                    # Store diagnostic data
                    for num, score in pattern_scores.most_common(10):
                        diagnostic_data.append({
                            'date': date_str,
                            'slot': slot_name,
                            'number': num,
                            'score_total': score,
                            'in_hot': 1 if num in self.SLOT_HOT[slot_name] else 0,
                            'in_global': 1 if num in self.GLOBAL_MULTI_SLOT_HOT else 0,
                            'digit_bonus': self.digit_bonus(slot_name, num),
                            'opposite_bonus': self.cross_slot_opp_bonus(slot_name, num)
                        })
                    
                    pred_data = {
                        'numbers': final_pred,
                        'top_k': top_k,
                        'confidence': pattern_scores.most_common(1)[0][1] if pattern_scores else 0
                    }
                
                predictions[date_str][slot_name] = pred_data
        
        # Save enhanced diagnostic data
        if diagnostic_data:
            diag_df = pd.DataFrame(diagnostic_data)
            diag_path = os.path.join(self.predictions_dir, 'scr10_diagnostic.xlsx')
            diag_df.to_excel(diag_path, index=False)
        
        return predictions

    def get_base_strategies(self, numbers):
        """Get base prediction strategies"""
        base_scores = Counter()
        
        # Multiple strategy scores
        strategies = {
            'frequency': self.frequency_strategy(numbers),
            'gap': self.gap_strategy(numbers),
            'pattern': self.pattern_strategy(numbers),
            'markov': self.markov_strategy(numbers)
        }
        
        # Combine strategies
        for strategy_name, scores in strategies.items():
            for num, score in scores.items():
                base_scores[num] += score
        
        return base_scores

    def frequency_strategy(self, numbers):
        """Frequency-based strategy"""
        freq = Counter(numbers[-30:])
        total = sum(freq.values())
        return {num: count/total for num, count in freq.items()}

    def gap_strategy(self, numbers):
        """Gap-based strategy"""
        last_seen = {}
        for i, num in enumerate(numbers):
            last_seen[num] = i
        
        current_idx = len(numbers) - 1
        gap_scores = {}
        
        for num in range(100):
            if num in last_seen:
                gap = current_idx - last_seen[num]
                gap_scores[num] = min(gap / 50, 1.0)  # Normalize
            else:
                gap_scores[num] = 0.5  # Neutral for unseen
        
        return gap_scores

    def pattern_strategy(self, numbers):
        """Pattern-based strategy"""
        if len(numbers) < 10:
            return {}
        
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
        
        pattern_scores = {}
        if sequences:
            last_seq = sequences[-1]
            if len(last_seq) >= 3:
                # Predict continuation
                avg_diff = (last_seq[-1] - last_seq[0]) / (len(last_seq) - 1)
                next_num = int((last_seq[-1] + avg_diff) % 100)
                pattern_scores[next_num] = 1.0
                
                # Also score numbers close to prediction
                for offset in [-3, -2, -1, 1, 2, 3]:
                    near_num = (next_num + offset) % 100
                    pattern_scores[near_num] = 0.5
        
        return pattern_scores

    def markov_strategy(self, numbers):
        """Markov chain strategy"""
        if len(numbers) < 10:
            return {}
        
        transitions = {}
        for i in range(1, len(numbers)):
            prev = numbers[i-1]
            curr = numbers[i]
            
            if prev not in transitions:
                transitions[prev] = {}
            transitions[prev][curr] = transitions[prev].get(curr, 0) + 1
        
        markov_scores = {}
        if numbers and numbers[-1] in transitions:
            next_probs = transitions[numbers[-1]]
            total = sum(next_probs.values())
            for num, count in next_probs.items():
                markov_scores[num] = count / total
        
        return markov_scores

    def integrate_online_predictions(self, online_predictions, slot_name):
        """Integrate online predictions"""
        online_scores = Counter()
        
        # Process online predictions (placeholder implementation)
        # In real implementation, parse the online data structure
        for source, predictions in online_predictions.items():
            if slot_name in predictions:
                for rank, num in enumerate(predictions[slot_name][:5], 1):
                    weight = 1.0 / rank  # Higher rank = more weight
                    online_scores[num] += weight
        
        return online_scores

    def smart_top_k_selection(self, scores, slot_name):
        """Smart top-k selection based on score distribution"""
        if not scores:
            return 10
        
        top_scores = [score for _, score in scores.most_common(10)]
        if not top_scores:
            return 10
        
        avg_score = np.mean(top_scores)
        max_score = max(top_scores)
        
        # Calculate confidence ratio
        confidence = max_score / avg_score if avg_score > 0 else 0
        
        if confidence > 2.0:
            return 5   # High confidence
        elif confidence > 1.5:
            return 10  # Medium confidence
        else:
            return 15  # Low confidence

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

    def generate_predictions(self, df, days=3):
        """Generate ultimate predictions"""
        print("🚀 Running SCR10 Ultimate Engine...")
        ultimate_pred = self.advanced_prediction_engine(df, days)
        
        # Convert to DataFrame
        predictions_list = []
        for date, slots in ultimate_pred.items():
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
        """Create comprehensive output"""
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
        
        # Save files with new paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_path = os.path.join(self.predictions_dir, f'scr10_predictions_{timestamp}.xlsx')
        detailed_path = os.path.join(self.predictions_dir, f'scr10_detailed_{timestamp}.xlsx')
        analysis_path = os.path.join(self.predictions_dir, f'scr10_analysis_{timestamp}.txt')
        diagnostic_path = os.path.join(self.predictions_dir, 'scr10_diagnostic.xlsx')
        
        wide_df.to_excel(predictions_path, index=False)
        predictions_df.to_excel(detailed_path, index=False)
        
        # Create pattern analysis report
        self.create_pattern_report(predictions_df, df, analysis_path)
        
        # Calculate relative paths for display
        rel_predictions = os.path.relpath(predictions_path, self.base_dir)
        rel_detailed = os.path.relpath(detailed_path, self.base_dir)
        rel_analysis = os.path.relpath(analysis_path, self.base_dir)
        rel_diagnostic = os.path.relpath(diagnostic_path, self.base_dir)
        rel_performance = os.path.relpath(self.performance_log, self.base_dir)
        
        print("\n✅ Files saved:")
        print(f"   - {rel_predictions}")
        print(f"   - {rel_detailed}")
        print(f"   - {rel_diagnostic}")
        print(f"   - {rel_analysis}")
        print(f"   - {rel_performance}")
        
        return wide_df

    def create_pattern_report(self, predictions_df, df, filename):
        """Create pattern analysis report"""
        with open(filename, 'w') as f:
            f.write("SCR10 ULTIMATE PREDICTOR - PATTERN ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PATTERN PACKS USED:\n")
            f.write(f"• Slot HOT packs: {self.SLOT_HOT}\n")
            f.write(f"• Global multi-slot HOT: {self.GLOBAL_MULTI_SLOT_HOT}\n")
            f.write(f"• Digit bias patterns: {self.DIGIT_BIAS}\n\n")
            
            f.write("ADVANCED FEATURES:\n")
            f.write("• Pattern-based scoring engine\n")
            f.write("• HOT/COLD number weighting\n")
            f.write("• Digit pattern bias integration\n")
            f.write("• Cross-slot opposite analysis\n")
            f.write("• Online source integration ready\n")
            f.write("• Range diversity enforcement\n")
            f.write("• Smart top-k selection\n\n")
            
            # Add predictions summary
            if not predictions_df.empty:
                f.write("PREDICTIONS SUMMARY:\n")
                for date in predictions_df['date'].unique():
                    f.write(f"\n{date}:\n")
                    date_pred = predictions_df[predictions_df['date'] == date]
                    for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                        slot_pred = date_pred[date_pred['slot'] == slot]
                        if not slot_pred.empty:
                            top_k = slot_pred['top_k'].iloc[0]
                            numbers = slot_pred['number'].tolist()[:5]
                            f.write(f"  {slot} (Top-{top_k}): {', '.join(numbers)}\n")

def main():
    organize_old_scr10_files()
    print("🧹 Organizing SCR10 files into structured folders...")
    
    print("=== SCR10 ULTIMATE PREDICTOR ===")
    print("🎯 Pattern Packs + Online Integration + Smart Scoring")
    
    predictor = SCR10UltimatePredictor()
    file_path = 'number prediction learn.xlsx'
    
    # Load data
    df = predictor.load_data(file_path)
    
    if df is not None and len(df) > 0:
        print(f"📊 Historical data: {len(df)} records")
        print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Generate predictions
        print("\n🎯 Generating SCR10 ultimate predictions...")
        predictions = predictor.generate_predictions(df, days=3)
        
        # Create output
        wide_predictions = predictor.create_output(predictions, df)
        
        # Display predictions
        if len(wide_predictions) > 0:
            first_date = wide_predictions['date'].iloc[0]
            print(f"\n🎲 SCR10 PREDICTIONS FOR {first_date}:")
            
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in wide_predictions.columns:
                    numbers = wide_predictions[wide_predictions['date'] == first_date][slot_name].iloc[0]
                    top_k = wide_predictions[wide_predictions['date'] == first_date][f'{slot_name}_top_k'].iloc[0]
                    print(f"   {slot_name} (Top-{top_k}): {numbers}")
            
            print(f"\n🔄 PATTERN ANALYSIS:")
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in wide_predictions.columns:
                    numbers_str = wide_predictions[wide_predictions['date'] == first_date][slot_name].iloc[0]
                    numbers = [int(x) for x in numbers_str.split(', ')]
                    
                    # Show pattern matches
                    hot_matches = [n for n in numbers[:5] if n in predictor.SLOT_HOT[slot_name]]
                    global_matches = [n for n in numbers[:5] if n in predictor.GLOBAL_MULTI_SLOT_HOT]
                    
                    print(f"   {slot_name}: HOT={hot_matches}, GLOBAL={global_matches}")
        
        print("\n🔬 SCR10 METHODOLOGY:")
        print("   • Pattern packs integration (HOT/COLD numbers)")
        print("   • Digit bias analysis (tens/ones/sums)")
        print("   • Global multi-slot HOT numbers")
        print("   • Cross-slot opposite pattern detection")
        print("   • Online source integration ready")
        print("   • Range diversity enforcement")
        print("   • Enhanced diagnostic scoring")
        
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()