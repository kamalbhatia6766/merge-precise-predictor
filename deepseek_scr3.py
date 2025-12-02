import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import Counter, defaultdict
import math
from pathlib import Path
from quant_excel_loader import load_results_excel

warnings.filterwarnings('ignore')

class HybridPredictor:
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

    def ensemble_prediction(self, numbers, top_k=10):
        """Hybrid ensemble combining multiple strategies"""
        if len(numbers) < 10:
            return self.fallback_prediction(numbers, top_k)
        
        predictions = {}
        
        # Strategy 1: Frequency + Recency (Original script approach)
        freq_pred = self.frequency_recency_analysis(numbers, top_k)
        predictions['freq_recency'] = (freq_pred, 0.30)
        
        # Strategy 2: Gap Analysis (Enhanced script approach)
        gap_pred = self.gap_analysis_prediction(numbers, top_k)
        predictions['gap_analysis'] = (gap_pred, 0.25)
        
        # Strategy 3: Pattern Sequences
        pattern_pred = self.pattern_sequence_analysis(numbers, top_k)
        predictions['patterns'] = (pattern_pred, 0.20)
        
        # Strategy 4: Markov Transitions
        markov_pred = self.markov_chain_prediction(numbers, top_k)
        predictions['markov'] = (markov_pred, 0.15)
        
        # Strategy 5: Hot/Cold Balance
        hotcold_pred = self.hot_cold_analysis(numbers, top_k)
        predictions['hot_cold'] = (hotcold_pred, 0.10)
        
        # Combine all strategies
        combined_scores = defaultdict(float)
        
        for strategy, (preds, weight) in predictions.items():
            for rank, num in enumerate(preds):
                # Higher ranked predictions get more weight
                position_weight = (len(preds) - rank) / len(preds)
                combined_scores[num] += weight * position_weight
        
        # Get top predictions
        final_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in final_predictions[:top_k]]

    def frequency_recency_analysis(self, numbers, top_k):
        """Original script's frequency analysis with exponential decay"""
        if len(numbers) < 30:
            window = len(numbers)
        else:
            window = 30
            
        recent_data = numbers[-window:]
        weights = np.exp(np.linspace(0, 1, window))
        weights = weights / weights.sum()
        
        number_counts = {}
        for idx, num in enumerate(recent_data):
            weight = weights[idx] if idx < len(weights) else 1.0
            number_counts[num] = number_counts.get(num, 0) + weight
        
        hot_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in hot_numbers[:top_k]]

    def gap_analysis_prediction(self, numbers, top_k):
        """Enhanced script's gap analysis"""
        positions = {}
        for i, num in enumerate(numbers):
            if num not in positions:
                positions[num] = []
            positions[num].append(i)
        
        gap_scores = {}
        current_idx = len(numbers) - 1
        
        for num in range(100):
            if num in positions and len(positions[num]) > 1:
                gaps = [positions[num][i] - positions[num][i-1] for i in range(1, len(positions[num]))]
                avg_gap = np.mean(gaps)
                current_gap = current_idx - positions[num][-1]
                # Higher score for numbers that are more "due"
                gap_scores[num] = current_gap / avg_gap if avg_gap > 0 else 10.0
            else:
                gap_scores[num] = 10.0  # Never seen or seen once
        
        due_numbers = sorted(gap_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in due_numbers[:top_k]]

    def pattern_sequence_analysis(self, numbers, top_k):
        """Detect and use number sequences"""
        sequences = defaultdict(list)
        
        # Look for sequences of length 2-4
        for length in range(2, 5):
            for i in range(len(numbers) - length):
                seq = tuple(numbers[i:i+length])
                next_val = numbers[i+length]
                sequences[seq].append(next_val)
        
        # Use the most recent sequence to predict
        if len(numbers) >= 3:
            recent_seq = tuple(numbers[-3:])
            if recent_seq in sequences:
                next_vals = sequences[recent_seq]
                counter = Counter(next_vals)
                return [num for num, count in counter.most_common(top_k)]
        
        return self.fallback_prediction(numbers, top_k)

    def markov_chain_prediction(self, numbers, top_k):
        """Markov chain transition probabilities"""
        transitions = {}
        
        for i in range(1, len(numbers)):
            prev = numbers[i-1]
            curr = numbers[i]
            
            if prev not in transitions:
                transitions[prev] = {}
            transitions[prev][curr] = transitions[prev].get(curr, 0) + 1
        
        # Predict from last number
        if numbers and numbers[-1] in transitions:
            next_probs = transitions[numbers[-1]]
            likely_next = sorted(next_probs.items(), key=lambda x: x[1], reverse=True)
            return [num for num, count in likely_next[:top_k]]
        
        return self.fallback_prediction(numbers, top_k)

    def hot_cold_analysis(self, numbers, top_k):
        """Balance between hot and cold numbers"""
        if len(numbers) < 20:
            return self.fallback_prediction(numbers, top_k)
        
        # Hot numbers (frequent recently)
        hot_window = min(30, len(numbers))
        hot_data = numbers[-hot_window:]
        hot_freq = Counter(hot_data)
        hot_numbers = [num for num, count in hot_freq.most_common(top_k//2)]
        
        # Cold numbers (due to appear based on full history)
        full_freq = Counter(numbers)
        avg_frequency = len(numbers) / 100
        cold_candidates = []
        
        for num in range(100):
            actual_freq = full_freq.get(num, 0)
            if actual_freq < avg_frequency * 0.7:  # Less frequent than average
                cold_candidates.append(num)
        
        # Take top cold candidates
        cold_numbers = cold_candidates[:top_k//2]
        
        return hot_numbers + cold_numbers

    def fallback_prediction(self, numbers, top_k):
        """Fallback when data is insufficient"""
        if not numbers:
            return list(range(top_k))
        
        freq = Counter(numbers)
        return [num for num, count in freq.most_common(top_k)]

    def generate_hybrid_predictions(self, df, days=3, top_k=5):
        """Generate predictions using hybrid approach"""
        predictions = []
        
        for day_offset in range(1, days + 1):
            target_date = datetime.now().date() + timedelta(days=day_offset)
            
            for slot in [1, 2, 3, 4]:
                slot_data = df[df['slot'] == slot]
                numbers = slot_data['number'].tolist()
                
                if len(numbers) < 5:
                    pred_numbers = self.fallback_prediction(numbers, top_k)
                else:
                    pred_numbers = self.ensemble_prediction(numbers, top_k * 2)  # Get more for filtering
                
                # Apply final filtering for diversity
                final_pred = self.apply_smart_filter(pred_numbers, top_k)
                
                for rank, number in enumerate(final_pred, 1):
                    predictions.append({
                        'date': target_date.strftime('%Y-%m-%d'),
                        'slot': self.slot_names[slot],
                        'rank': rank,
                        'number': f"{number:02d}"
                    })
        
        return pd.DataFrame(predictions)

    def apply_smart_filter(self, predictions, top_k):
        """Smart filtering for balanced predictions"""
        if len(predictions) <= top_k:
            return predictions[:top_k]
        
        # Ensure diversity across ranges
        range_groups = {
            'low': [n for n in predictions if 0 <= n <= 33],
            'medium': [n for n in predictions if 34 <= n <= 66],
            'high': [n for n in predictions if 67 <= n <= 99]
        }
        
        selected = []
        
        # Take best from each range
        for range_name in ['low', 'medium', 'high']:
            if range_groups[range_name]:
                selected.append(range_groups[range_name][0])
        
        # Fill remaining with highest confidence
        remaining = top_k - len(selected)
        if remaining > 0:
            for pred in predictions:
                if pred not in selected and len(selected) < top_k:
                    selected.append(pred)
        
        return selected[:top_k]

    def create_output_files(self, predictions_df):
        """Create all output files in SCR3-specific directory"""
        # Create SCR3 directory structure
        BASE_DIR = Path(__file__).resolve().parent
        SCR3_DIR = BASE_DIR / "predictions" / "deepseek_scr3"
        SCR3_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Wide format predictions
        wide_df = predictions_df.pivot_table(
            index='date', 
            columns='slot', 
            values='number',
            aggfunc=lambda x: ', '.join(x)
        ).reset_index()
        
        column_order = ['date'] + [self.slot_names[i] for i in [1, 2, 3, 4]]
        wide_df = wide_df.reindex(columns=column_order)
        
        # Save timestamped files
        pred_path = SCR3_DIR / f"scr3_predictions_{timestamp}.xlsx"
        detail_path = SCR3_DIR / f"scr3_detailed_{timestamp}.xlsx"
        
        wide_df.to_excel(pred_path, index=False)
        predictions_df.to_excel(detail_path, index=False)
        
        # Save latest copies for convenience
        wide_df.to_excel(SCR3_DIR / "scr3_predictions_latest.xlsx", index=False)
        predictions_df.to_excel(SCR3_DIR / "scr3_detailed_latest.xlsx", index=False)
        
        return wide_df, pred_path, detail_path

def main():
    print("=== HYBRID NUMBER PREDICTOR ===")
    print("🎯 Combining Best of Both Approaches")
    
    predictor = HybridPredictor()
    file_path = 'number prediction learn.xlsx'
    
    # Load data
    df = predictor.load_data(file_path)
    
    if df is not None and len(df) > 0:
        print(f"✅ Data loaded: {len(df)} records")
        print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Show data summary
        print("\n📊 Data Summary:")
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            print(f"  {predictor.slot_names[slot]}: {len(slot_data)} records")
        
        # Generate predictions
        print("\n🎯 Generating hybrid predictions...")
        predictions = predictor.generate_hybrid_predictions(df, days=3, top_k=5)
        
        # Create output files
        wide_predictions, pred_path, detail_path = predictor.create_output_files(predictions)
        
        print("✅ Hybrid predictions generated successfully!")
        print("💾 Files saved:")
        print(f"   - {pred_path}")
        print(f"   - {detail_path}")
        
        # Display tomorrow's predictions
        if len(wide_predictions) > 0:
            print("\n🎲 HYBRID PREDICTIONS FOR TOMORROW:")
            tomorrow_pred = wide_predictions.iloc[0]
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in tomorrow_pred:
                    print(f"   {slot_name}: {tomorrow_pred[slot_name]}")
        
        print("\n🔍 Strategy: Combined frequency, gap analysis, patterns, Markov chains, and hot/cold balance")
        
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()