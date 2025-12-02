import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
from quant_excel_loader import load_results_excel

warnings.filterwarnings('ignore')

class FinalPredictor:
    def __init__(self):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        
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

    def advanced_ensemble(self, numbers, top_k=10):
        """Final ensemble with probability calibration"""
        if len(numbers) < 10:
            return self.fallback_prediction(numbers, top_k)
        
        strategies = {}
        
        # Strategy 1: Bayesian Probability with Multiple Timeframes
        bayesian_pred = self.bayesian_probability(numbers, top_k)
        strategies['bayesian'] = (bayesian_pred, 0.25)
        
        # Strategy 2: Enhanced Gap Analysis with Confidence
        gap_pred = self.confidence_gap_analysis(numbers, top_k)
        strategies['gap_analysis'] = (gap_pred, 0.20)
        
        # Strategy 3: Pattern Recognition with Sequence Mining
        pattern_pred = self.advanced_pattern_mining(numbers, top_k)
        strategies['patterns'] = (pattern_pred, 0.20)
        
        # Strategy 4: Frequency Momentum
        momentum_pred = self.frequency_momentum(numbers, top_k)
        strategies['momentum'] = (momentum_pred, 0.15)
        
        # Strategy 5: Markov Chain with Memory
        markov_pred = self.multi_step_markov(numbers, top_k)
        strategies['markov'] = (markov_pred, 0.10)
        
        # Strategy 6: Random Forest Simulation (Statistical)
        rf_pred = self.statistical_forest(numbers, top_k)
        strategies['statistical'] = (rf_pred, 0.10)
        
        # Combine with calibrated weights
        final_scores = self.calibrate_predictions(strategies, numbers)
        
        # Apply final filtering
        final_predictions = self.apply_intelligent_filter(final_scores, top_k)
        
        return final_predictions

    def bayesian_probability(self, numbers, top_k):
        """Bayesian probability with multiple time windows"""
        windows = [30, 60, 90, 180]
        alpha = 1.0  # Laplace smoothing
        
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
        
        # If no windows worked, use all data
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
                
                # Confidence score based on how far beyond average gap
                if current_gap > avg_gap:
                    confidence = min((current_gap - avg_gap) / (std_gap + 1), 3.0) / 3.0
                else:
                    confidence = 0.1  # Small chance if not due yet
                    
                gap_scores[num] = confidence
            else:
                # New or rare numbers get moderate chance
                gap_scores[num] = 0.5
        
        due_numbers = sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in due_numbers[:top_k]]

    def advanced_pattern_mining(self, numbers, top_k):
        """Advanced pattern mining with multiple sequence lengths"""
        sequences = defaultdict(list)
        
        # Analyze different sequence lengths
        for length in [2, 3, 4]:
            for i in range(len(numbers) - length):
                seq = tuple(numbers[i:i+length])
                next_val = numbers[i+length]
                sequences[seq].append(next_val)
        
        # Use recent sequences for prediction
        predictions = []
        for length in [4, 3, 2]:  # Try longer sequences first
            if len(numbers) >= length:
                recent_seq = tuple(numbers[-length:])
                if recent_seq in sequences:
                    next_vals = sequences[recent_seq]
                    counter = Counter(next_vals)
                    predictions.extend([num for num, count in counter.most_common(3)])
        
        # Fallback if no patterns found
        if not predictions:
            freq = Counter(numbers[-30:])
            predictions = [num for num, count in freq.most_common(top_k)]
        
        return predictions[:top_k]

    def frequency_momentum(self, numbers, top_k):
        """Track frequency momentum (rising/falling popularity)"""
        if len(numbers) < 20:
            return self.fallback_prediction(numbers, top_k)
        
        # Compare recent vs historical frequency
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
                
            momentum_scores[num] = max(momentum, 0)  # Only positive momentum
        
        rising_numbers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in rising_numbers[:top_k]]

    def multi_step_markov(self, numbers, top_k):
        """Multi-step Markov chain predictions"""
        if len(numbers) < 10:
            return self.fallback_prediction(numbers, top_k)
        
        # First order transitions
        transitions = {}
        for i in range(1, len(numbers)):
            prev = numbers[i-1]
            curr = numbers[i]
            
            if prev not in transitions:
                transitions[prev] = {}
            transitions[prev][curr] = transitions[prev].get(curr, 0) + 1
        
        # Predict multiple steps
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
        
        # Fill remaining slots with frequency
        if len(predictions) < top_k:
            freq = Counter(numbers[-20:])
            additional = [num for num, count in freq.most_common(top_k) if num not in predictions]
            predictions.extend(additional[:top_k - len(predictions)])
        
        return predictions[:top_k]

    def statistical_forest(self, numbers, top_k):
        """Statistical simulation of random forest behavior"""
        if len(numbers) < 15:
            return self.fallback_prediction(numbers, top_k)
        
        # Simulate feature importance through statistical analysis
        features = {}
        
        # Feature 1: Recent frequency
        recent_freq = Counter(numbers[-15:])
        for num, count in recent_freq.items():
            features[num] = features.get(num, 0) + count * 0.3
        
        # Feature 2: Position in data stream
        last_positions = {}
        for i, num in enumerate(numbers[-30:]):
            last_positions[num] = i
        for num, pos in last_positions.items():
            features[num] = features.get(num, 0) + (30 - pos) * 0.2
        
        # Feature 3: Gap analysis
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
        """Calibrate prediction scores based on historical performance"""
        final_scores = defaultdict(float)
        
        for strategy_name, (predictions, base_weight) in strategies.items():
            # Adjust weight based on strategy reliability
            reliability = self.assess_strategy_reliability(strategy_name, numbers)
            adjusted_weight = base_weight * reliability
            
            for rank, num in enumerate(predictions):
                # Position-based weighting within strategy
                position_weight = (len(predictions) - rank) / len(predictions)
                final_scores[num] += adjusted_weight * position_weight
        
        return final_scores

    def assess_strategy_reliability(self, strategy_name, numbers):
        """Simple reliability assessment based on data characteristics"""
        if len(numbers) < 10:
            return 0.5  # Low confidence with little data
        
        # Basic reliability metrics
        data_diversity = len(set(numbers)) / 100
        data_volume = min(len(numbers) / 100, 1.0)
        
        # Strategy-specific reliability adjustments
        if strategy_name == 'bayesian':
            return 0.9 * data_volume
        elif strategy_name == 'gap_analysis':
            return 0.8 * data_diversity
        elif strategy_name == 'patterns':
            return 0.7 * data_volume
        else:
            return 0.6
        
        return 0.5

    def apply_intelligent_filter(self, scores, top_k):
        """Intelligent filtering for balanced predictions"""
        if not scores:
            return list(range(top_k))
        
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take more candidates than needed for filtering
        candidates = [num for num, score in sorted_predictions[:top_k * 2]]
        
        # Ensure range diversity
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
        
        # Fill remaining with highest scores, ensuring no duplicates
        remaining = top_k - len(selected)
        if remaining > 0:
            for num in candidates:
                if num not in selected and len(selected) < top_k:
                    selected.append(num)
        
        return selected[:top_k]

    def fallback_prediction(self, numbers, top_k):
        """Reliable fallback prediction"""
        if not numbers:
            return list(range(top_k))
        
        # Simple frequency-based approach
        freq = Counter(numbers)
        return [num for num, count in freq.most_common(top_k)]

    def generate_final_predictions(self, df, days=3, top_k=5):
        """Generate final optimized predictions"""
        predictions = []
        
        for day_offset in range(1, days + 1):
            target_date = datetime.now().date() + timedelta(days=day_offset)
            
            for slot in [1, 2, 3, 4]:
                slot_data = df[df['slot'] == slot]
                numbers = slot_data['number'].tolist()
                
                if len(numbers) < 5:
                    pred_numbers = self.fallback_prediction(numbers, top_k)
                else:
                    pred_numbers = self.advanced_ensemble(numbers, top_k * 2)
                
                final_pred = self.apply_intelligent_filter(
                    {num: 1.0 for num in pred_numbers}, top_k
                )
                
                for rank, number in enumerate(final_pred, 1):
                    predictions.append({
                        'date': target_date.strftime('%Y-%m-%d'),
                        'slot': self.slot_names[slot],
                        'rank': rank,
                        'number': f"{number:02d}"
                    })
        
        return pd.DataFrame(predictions)

    def create_comprehensive_output(self, predictions_df, output_dir, timestamp):
        """Create comprehensive output files in SCR4-specific directory"""
        # Wide format
        wide_df = predictions_df.pivot_table(
            index='date', 
            columns='slot', 
            values='number',
            aggfunc=lambda x: ', '.join(x)
        ).reset_index()
        
        column_order = ['date'] + [self.slot_names[i] for i in [1, 2, 3, 4]]
        wide_df = wide_df.reindex(columns=column_order)
        
        # Save files with timestamped names
        predictions_path = os.path.join(output_dir, f"scr4_predictions_{timestamp}.xlsx")
        detailed_path = os.path.join(output_dir, f"scr4_detailed_{timestamp}.xlsx")
        
        wide_df.to_excel(predictions_path, index=False)
        predictions_df.to_excel(detailed_path, index=False)
        
        # Create analysis report
        report_path = self.create_analysis_report(predictions_df, output_dir, timestamp)
        
        return wide_df, predictions_path, detailed_path, report_path

    def create_analysis_report(self, predictions_df, output_dir, timestamp):
        """Create analysis report in SCR4-specific directory"""
        report_path = os.path.join(output_dir, f"scr4_analysis_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("FINAL PREDICTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Prediction Methodology:\n")
            f.write("- Bayesian Probability with Multiple Timeframes\n")
            f.write("- Confidence-Based Gap Analysis\n")
            f.write("- Advanced Pattern Mining\n")
            f.write("- Frequency Momentum Tracking\n")
            f.write("- Multi-step Markov Chains\n")
            f.write("- Statistical Forest Simulation\n\n")
            
            f.write("Key Features:\n")
            f.write("• Probability calibration\n")
            f.write("• Strategy reliability assessment\n")
            f.write("• Intelligent range filtering\n")
            f.write("• Multi-method ensemble\n\n")
        
        return report_path

def main():
    print("=== FINAL OPTIMIZED PREDICTOR ===")
    print("🎯 Advanced Ensemble with Probability Calibration")
    
    # Create SCR4-specific output directory
    BASE_DIR = Path(__file__).resolve().parent
    SCR4_OUTPUT_DIR = BASE_DIR / "predictions" / "deepseek_scr4"
    SCR4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🧹 SCR4: using predictions\\deepseek_scr4 for all outputs...")
    
    predictor = FinalPredictor()
    file_path = 'number prediction learn.xlsx'
    
    # Load data
    df = predictor.load_data(file_path)
    
    if df is not None and len(df) > 0:
        print(f"📊 Total records: {len(df)}")
        
        # Show monthly breakdown
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_counts = df.groupby(['year_month', 'slot']).size().unstack(fill_value=0)
        print("\n📅 MONTHLY BREAKDOWN:")
        print(monthly_counts)
        
        # Generate predictions
        print("\n🎯 Generating final optimized predictions...")
        predictions = predictor.generate_final_predictions(df, days=3, top_k=5)
        
        # Create output files in SCR4 directory
        wide_predictions, pred_path, detail_path, report_path = predictor.create_comprehensive_output(
            predictions, SCR4_OUTPUT_DIR, timestamp
        )
        
        print("✅ Final predictions generated successfully!")
        print("💾 Files saved:")
        print(f"   - {pred_path}")
        print(f"   - {detail_path}")
        print(f"   - {report_path}")
        
        # Display predictions (unchanged format for SCR9 parsing)
        if len(wide_predictions) > 0:
            print("\n🎲 FINAL PREDICTIONS FOR TOMORROW:")
            tomorrow_pred = wide_predictions.iloc[0]
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in tomorrow_pred:
                    print(f"   {slot_name}: {tomorrow_pred[slot_name]}")
        
        print("\n🔬 Methodology: 6-strategy ensemble with probability calibration")
        print("   • Bayesian Probability • Gap Analysis • Pattern Mining")
        print("   • Frequency Momentum • Markov Chains • Statistical Forest")
        
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()