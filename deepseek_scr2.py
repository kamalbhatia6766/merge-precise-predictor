import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import Counter, defaultdict
import math
import os
import shutil
from quant_excel_loader import load_results_excel


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have a consistent 'date' column even if the Excel header is 'DATE'.
    - Keeps original columns.
    - Adds a lowercase 'date' alias as datetime.
    """
    cols_norm = [str(c).strip() for c in df.columns]
    df.columns = cols_norm

    # Try to find a date-like column
    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    elif 'DATE' in df.columns:
        date_col = 'DATE'

    if date_col is not None:
        # Create / overwrite a canonical lowercase 'date' column as datetime
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        # No date column at all – leave as is
        pass

    return df

warnings.filterwarnings('ignore')

class UltimateNumberPredictor:
    def __init__(self):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.predictions_dir = os.path.join(self.base_dir, "predictions", "deepseek_scr2")
        self.logs_dir = os.path.join(self.base_dir, "logs", "performance")
        
        # Create directories
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _ensure_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure df has columns ['date', 'slot', 'number'].
        If df already has 'slot' and 'number', return as-is.
        Otherwise, convert from wide format: DATE, FRBD, GZBD, GALI, DSWR.
        """
        slot_map = {name: sid for sid, name in self.slot_names.items()}

        # Already long-format?
        if 'slot' in df.columns and 'number' in df.columns:
            # Make sure 'date' exists
            if 'date' not in df.columns and 'DATE' in df.columns:
                df = df.copy()
                df['date'] = pd.to_datetime(df['DATE'], errors='coerce')

            df['slot'] = df['slot'].apply(lambda x: slot_map.get(x, x))
            df['number'] = pd.to_numeric(df['number'], errors='coerce')
            df = df.dropna(subset=['date', 'slot', 'number'])
            df['slot'] = df['slot'].astype(int)
            df['number'] = df['number'].astype(int) % 100
            return df[['date', 'slot', 'number']]

        wide_df = df.copy()

        # Normalize date column
        if 'date' not in wide_df.columns and 'DATE' in wide_df.columns:
            wide_df['date'] = pd.to_datetime(wide_df['DATE'], errors='coerce')

        slot_cols = [c for c in ['FRBD', 'GZBD', 'GALI', 'DSWR'] if c in wide_df.columns]
        if not slot_cols:
            raise ValueError("No slot columns found (expected FRBD, GZBD, GALI, DSWR).")

        long_parts = []
        for col in slot_cols:
            part = wide_df[['date', col]].copy()
            part = part.rename(columns={col: 'number'})
            part['slot'] = slot_map.get(col, col)
            long_parts.append(part)

        long_df = pd.concat(long_parts, ignore_index=True)

        # Clean number type
        long_df['number'] = pd.to_numeric(long_df['number'], errors='coerce')
        long_df = long_df.dropna(subset=['date', 'slot', 'number'])
        long_df['slot'] = long_df['slot'].astype(int)
        long_df['number'] = long_df['number'].astype(int) % 100

        return long_df[['date', 'slot', 'number']]
        
    def load_data(self, file_path):
        """
        Canonical data loader for this script.
        Uses the central quant_excel_loader so that the Excel structure
        only needs to be maintained in one place.
        """
        try:
            df = load_results_excel(file_path)
            print(f"✅ Data loaded successfully via quant_excel_loader: {len(df)} records")
            return df
        except Exception as e:
            print(f"❌ Error loading data via quant_excel_loader: {e}")
            raise
    
    def clean_number(self, x):
        """Convert to 2-digit number (00-99)"""
        if pd.isna(x):
            return None
        try:
            s = str(x).strip()
            digits = ''.join([c for c in s if c.isdigit()])
            if not digits:
                return None
            num = int(digits)
            return num % 100
        except Exception as e:
            return None

    # Advanced Pattern Recognition
    def analyze_frequency_patterns(self, numbers, lookback=90):
        """Advanced frequency analysis with multiple time windows"""
        if len(numbers) < 10:
            return self.simple_frequency(numbers)
        
        patterns = {}
        
        # Multiple lookback periods
        windows = [30, 60, 90, 180]
        for window in windows:
            if len(numbers) >= window:
                recent = numbers[-window:]
                freq = Counter(recent)
                patterns[f'freq_{window}'] = freq
        
        # Exponential weighted moving average
        weights = np.exp(np.linspace(0, 1, min(60, len(numbers))))
        weights = weights / weights.sum()
        weighted_freq = {}
        for i, num in enumerate(numbers[-60:]):
            weight = weights[i] if i < len(weights) else 1.0
            weighted_freq[num] = weighted_freq.get(num, 0) + weight
        patterns['weighted_freq'] = weighted_freq
        
        return patterns

    def simple_frequency(self, numbers):
        """Simple frequency count"""
        return {'simple_freq': Counter(numbers)}

    def digit_sum_analysis(self, numbers):
        """Analyze digit sum patterns"""
        digit_sums = [sum(int(d) for d in str(n).zfill(2)) for n in numbers]
        sum_freq = Counter(digit_sums)
        
        # Analyze sum ranges
        ranges = {
            'low_sums': [n for n in numbers if sum(int(d) for d in str(n).zfill(2)) <= 5],
            'medium_sums': [n for n in numbers if 6 <= sum(int(d) for d in str(n).zfill(2)) <= 12],
            'high_sums': [n for n in numbers if sum(int(d) for d in str(n).zfill(2)) >= 13]
        }
        
        return {
            'sum_frequencies': sum_freq,
            'sum_ranges': ranges
        }

    def range_distribution(self, numbers):
        """Analyze number range distribution"""
        ranges = {
            '0-24': [n for n in numbers if 0 <= n <= 24],
            '25-49': [n for n in numbers if 25 <= n <= 49],
            '50-74': [n for n in numbers if 50 <= n <= 74],
            '75-99': [n for n in numbers if 75 <= n <= 99]
        }
        
        range_freq = {k: len(v) for k, v in ranges.items()}
        return range_freq

    def time_based_patterns(self, df, slot):
        """Analyze day-of-week and monthly patterns"""
        slot_data = df[df['slot'] == slot].copy()
        
        # Day of week patterns
        slot_data['dow'] = slot_data['date'].dt.dayofweek
        dow_patterns = {}
        for day in range(7):
            day_data = slot_data[slot_data['dow'] == day]
            if len(day_data) > 0:
                dow_patterns[day] = Counter(day_data['number'])
        
        # Monthly patterns
        slot_data['month'] = slot_data['date'].dt.month
        month_patterns = {}
        for month in range(1, 13):
            month_data = slot_data[slot_data['month'] == month]
            if len(month_data) > 0:
                month_patterns[month] = Counter(month_data['number'])
        
        return {
            'dow_patterns': dow_patterns,
            'month_patterns': month_patterns
        }

    def gap_analysis_enhanced(self, numbers):
        """Enhanced gap analysis with probability scoring"""
        last_seen = {}
        gaps = {}
        positions = {}
        
        for i, num in enumerate(numbers):
            if num not in positions:
                positions[num] = []
            positions[num].append(i)
        
        # Calculate gap statistics
        gap_stats = {}
        for num in range(100):
            if num in positions and len(positions[num]) > 1:
                gaps_list = [positions[num][i] - positions[num][i-1] for i in range(1, len(positions[num]))]
                avg_gap = np.mean(gaps_list)
                std_gap = np.std(gaps_list)
                current_gap = len(numbers) - positions[num][-1]
                
                gap_stats[num] = {
                    'avg_gap': avg_gap,
                    'std_gap': std_gap,
                    'current_gap': current_gap,
                    'due_score': current_gap / avg_gap if avg_gap > 0 else 10.0
                }
            else:
                gap_stats[num] = {
                    'avg_gap': 999,
                    'std_gap': 0,
                    'current_gap': len(numbers),
                    'due_score': 10.0
                }
        
        return gap_stats

    def sequence_detection(self, numbers, min_length=3):
        """Detect number sequences and patterns"""
        sequences = []
        current_seq = [numbers[0]]
        
        for i in range(1, len(numbers)):
            diff = numbers[i] - numbers[i-1]
            if abs(diff) <= 3:  # Small difference for sequence
                current_seq.append(numbers[i])
            else:
                if len(current_seq) >= min_length:
                    sequences.append(current_seq)
                current_seq = [numbers[i]]
        
        if len(current_seq) >= min_length:
            sequences.append(current_seq)
        
        return sequences

    def markov_chain_analysis(self, numbers):
        """Simple Markov chain for transitions"""
        transitions = {}
        
        for i in range(1, len(numbers)):
            prev = numbers[i-1]
            curr = numbers[i]
            
            if prev not in transitions:
                transitions[prev] = {}
            transitions[prev][curr] = transitions[prev].get(curr, 0) + 1
        
        # Convert to probabilities
        for prev in transitions:
            total = sum(transitions[prev].values())
            for curr in transitions[prev]:
                transitions[prev][curr] = transitions[prev][curr] / total
        
        return transitions

    def ensemble_scoring(self, df, slot, top_k=10):
        """Combine multiple analysis methods for scoring"""
        df_long = self._ensure_long_format(df)

        # slot yahan numeric id (1–4) maana jayega.
        # Agar kabhi future mein slot name aa jaye (e.g. "FRBD"), to usko id mein map kar do.
        if isinstance(slot, int):
            slot_id = slot
        else:
            # Reverse map: "FRBD" -> 1, "GZBD" -> 2, etc.
            name_to_id = {name: sid for sid, name in self.slot_names.items()}
            slot_id = name_to_id.get(slot, slot)

        slot_data = df_long[df_long['slot'] == slot_id]
        numbers = slot_data['number'].tolist()

        if len(numbers) < 10:
            return self.fallback_scoring(numbers, top_k)
        
        # Get various analyses
        freq_patterns = self.analyze_frequency_patterns(numbers)
        digit_analysis = self.digit_sum_analysis(numbers)
        range_dist = self.range_distribution(numbers)
        time_patterns = self.time_based_patterns(df_long, slot_id)
        gap_stats = self.gap_analysis_enhanced(numbers)
        transitions = self.markov_chain_analysis(numbers)
        
        # Score each number
        scores = {}
        
        for num in range(100):
            score = 0
            reasons = []
            
            # 1. Frequency scoring (40%)
            if 'weighted_freq' in freq_patterns and num in freq_patterns['weighted_freq']:
                freq_score = freq_patterns['weighted_freq'][num]
                score += freq_score * 0.4
                reasons.append(f"Frequency: {freq_score:.3f}")
            
            # 2. Gap analysis scoring (25%)
            if num in gap_stats:
                due_score = min(gap_stats[num]['due_score'], 3.0) / 3.0  # Normalize to 0-1
                score += due_score * 0.25
                reasons.append(f"Due score: {due_score:.3f}")
            
            # 3. Transition probability (15%)
            last_number = numbers[-1] if numbers else None
            if last_number is not None and last_number in transitions and num in transitions[last_number]:
                trans_prob = transitions[last_number][num]
                score += trans_prob * 0.15
                reasons.append(f"Transition: {trans_prob:.3f}")
            
            # 4. Digit pattern (10%)
            digit_sum = sum(int(d) for d in str(num).zfill(2))
            if digit_sum in digit_analysis['sum_frequencies']:
                sum_freq = digit_analysis['sum_frequencies'][digit_sum] / len(numbers)
                score += sum_freq * 0.1
                reasons.append(f"Digit sum: {sum_freq:.3f}")
            
            # 5. Range distribution (10%)
            if 0 <= num <= 24 and range_dist['0-24'] > 0:
                range_score = range_dist['0-24'] / len(numbers)
                score += range_score * 0.025
            elif 25 <= num <= 49 and range_dist['25-49'] > 0:
                range_score = range_dist['25-49'] / len(numbers)
                score += range_score * 0.025
            elif 50 <= num <= 74 and range_dist['50-74'] > 0:
                range_score = range_dist['50-74'] / len(numbers)
                score += range_score * 0.025
            elif 75 <= num <= 99 and range_dist['75-99'] > 0:
                range_score = range_dist['75-99'] / len(numbers)
                score += range_score * 0.025
            
            if score > 0:
                scores[num] = (score, reasons)
        
        # Normalize scores
        if scores:
            max_score = max(scores.values(), key=lambda x: x[0])[0]
            if max_score > 0:
                scores = {num: (score/max_score, reasons) for num, (score, reasons) in scores.items()}
        
        # Return top predictions
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        return [(num, score, reasons) for num, (score, reasons) in sorted_predictions]

    def fallback_scoring(self, numbers, top_k):
        """Fallback scoring when data is insufficient"""
        if not numbers:
            return [(i, 0.1, ["Fallback"]) for i in range(top_k)]
        
        freq = Counter(numbers)
        predictions = []
        
        for num in range(100):
            score = freq.get(num, 0) / len(numbers) if numbers else 0.01
            predictions.append((num, score, ["Frequency fallback"]))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_k]

    def generate_predictions(self, df, days=3, top_k=5):
        """Generate predictions using ensemble methods"""
        predictions = []

        latest_data_date = df['date'].max().date()
        start_date = latest_data_date + timedelta(days=1)

        for day_offset in range(days):
            target_date = start_date + timedelta(days=day_offset)
            
            for slot in [1, 2, 3, 4]:
                slot_pred = self.ensemble_scoring(df, slot, top_k * 2)  # Get more for filtering
                
                # Apply diversity filter
                filtered_pred = self.apply_diversity_filter(slot_pred, top_k)
                
                for rank, (number, confidence, reasons) in enumerate(filtered_pred, 1):
                    predictions.append({
                        'date': target_date.strftime('%Y-%m-%d'),
                        'slot': self.slot_names[slot],
                        'rank': rank,
                        'number': f"{number:02d}",
                        'confidence': round(confidence, 3),
                        'reasons': '; '.join(reasons[:2])  # Top 2 reasons only
                    })
        
        return pd.DataFrame(predictions)

    def apply_diversity_filter(self, predictions, top_k):
        """Ensure prediction diversity across number ranges"""
        if len(predictions) <= top_k:
            return predictions[:top_k]
        
        # Group by tens digit
        tens_groups = defaultdict(list)
        for pred in predictions:
            tens_digit = pred[0] // 10
            tens_groups[tens_digit].append(pred)
        
        # Select diverse predictions
        selected = []
        used_tens = set()
        
        # First pass: take top from each tens group
        for tens in sorted(tens_groups.keys()):
            if tens_groups[tens]:
                selected.append(tens_groups[tens][0])
                used_tens.add(tens)
        
        # Fill remaining slots with highest confidence
        remaining_slots = top_k - len(selected)
        if remaining_slots > 0:
            all_preds = [p for p in predictions if p not in selected]
            all_preds.sort(key=lambda x: x[1], reverse=True)
            selected.extend(all_preds[:remaining_slots])
        
        return selected[:top_k]

    def create_prediction_sheet(self, predictions_df, output_format='wide'):
        """Create prediction sheet in desired format"""
        if output_format == 'wide':
            wide_df = predictions_df.pivot_table(
                index='date', 
                columns='slot', 
                values='number',
                aggfunc=lambda x: ', '.join(x)
            ).reset_index()
            
            # Reorder columns to match preferred order
            column_order = ['date'] + [self.slot_names[i] for i in [1, 2, 3, 4]]
            wide_df = wide_df.reindex(columns=column_order)
            
            return wide_df
        else:
            return predictions_df

    def generate_bet_plan(self, predictions_df):
        """Generate bet plan with stake allocation"""
        print("\n💰 GENERATING BET PLAN...")

        bet_data = []
        unique_dates = sorted(predictions_df['date'].unique())
        if not unique_dates:
            return pd.DataFrame()
        first_date_str = unique_dates[0]

        # Get predictions for the first available date
        tomorrow_pred = predictions_df[predictions_df['date'] == first_date_str]
        
        for slot in [1, 2, 3, 4]:
            slot_name = self.slot_names[slot]
            slot_preds = tomorrow_pred[tomorrow_pred['slot'] == slot_name]
            
            if len(slot_preds) == 0:
                continue
            
            # Determine ANDAR/BAHAR digits
            numbers = [int(pred['number']) for pred in slot_preds.to_dict('records')]
            andar_digit = self.calculate_andar_digit(numbers)
            bahar_digit = self.calculate_bahar_digit(numbers)
            
            # Assign tiers and stakes
            for i, (_, pred) in enumerate(slot_preds.iterrows()):
                if i < 2:  # Tier A
                    tier, stake = 'A', 20
                elif i < 4:  # Tier B
                    tier, stake = 'B', 10
                else:  # Tier C
                    tier, stake = 'C', 5

                bet_data.append({
                    'date': first_date_str,
                    'slot_name': slot_name,
                    'pick_rank': i + 1,
                    'number_2d': pred['number'],
                    'tier': tier,
                    'stake_rupees': stake,
                    'note': 'primary',
                    'andar_digit': andar_digit,
                    'bahar_digit': bahar_digit
                })
        
        bet_plan_df = pd.DataFrame(bet_data)
        return bet_plan_df

    def calculate_andar_digit(self, numbers):
        """Calculate ANDAR digit (most frequent tens digit)"""
        tens_digits = [n // 10 for n in numbers]
        return Counter(tens_digits).most_common(1)[0][0]

    def calculate_bahar_digit(self, numbers):
        """Calculate BAHAR digit (most frequent ones digit)"""
        ones_digits = [n % 10 for n in numbers]
        return Counter(ones_digits).most_common(1)[0][0]

# Alternative data loading method
def load_data_alternative(file_path):
    """
    Canonical data loader for this script.
    Uses the central quant_excel_loader so that the Excel structure
    only needs to be maintained in one place.
    """
    try:
        df = load_results_excel(file_path)
        print(f"✅ Data loaded successfully via quant_excel_loader: {len(df)} records")
        return df
    except Exception as e:
        print(f"❌ Error loading data via quant_excel_loader: {e}")
        raise

# Enhanced analysis
def enhanced_analysis(predictor, df):
    """Comprehensive data analysis"""
    print("\n=== ENHANCED DATA ANALYSIS ===")
    
    # Date range analysis
    print(f"\n📅 DATE RANGE: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Month-wise breakdown
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby(['year_month', 'slot']).size().unstack(fill_value=0)
    print("\n📊 MONTHLY RECORD COUNTS:")
    print(monthly_counts)
    
    # Advanced slot analysis
    for slot in [1, 2, 3, 4]:
        slot_data = df[df['slot'] == slot]
        numbers = slot_data['number'].tolist()
        
        if len(numbers) < 10:
            continue
            
        print(f"\n🎯 ADVANCED ANALYSIS - {predictor.slot_names[slot]}:")
        
        # Range distribution
        range_dist = predictor.range_distribution(numbers)
        print(f"  Range distribution: {range_dist}")
        
        # Digit sum analysis
        digit_analysis = predictor.digit_sum_analysis(numbers)
        common_sums = Counter(digit_analysis['sum_frequencies']).most_common(3)
        print(f"  Common digit sums: {common_sums}")
        
        # Gap analysis
        gap_stats = predictor.gap_analysis_enhanced(numbers)
        most_due = sorted(gap_stats.items(), key=lambda x: x[1]['due_score'], reverse=True)[:3]
        pairs = [(num, round(stats['due_score'], 2)) for num, stats in most_due]
        print(f"  Most due numbers (num, due_score_x): {pairs}")

def organize_old_scr2_files():
    """Move old SCR2 files from root to structured folders"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, "predictions", "deepseek_scr2")
    logs_dir = os.path.join(base_dir, "logs", "performance")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue

        lower = fname.lower()

        # Move SCR2 prediction files
        if (lower.startswith("ultimate_predictions") or 
            lower.startswith("ultimate_detailed") or 
            lower.startswith("advanced_predictions") or
            lower.startswith("advanced_detailed") or
            lower.startswith("detailed_predictions")) and lower.endswith(".xlsx"):
            shutil.move(fpath, os.path.join(pred_dir, f"scr2_{fname}"))
        elif lower == "bet_plan.xlsx":
            shutil.move(fpath, os.path.join(pred_dir, "scr2_bet_plan.xlsx"))
        elif lower.startswith("analysis_report") and lower.endswith(".txt"):
            shutil.move(fpath, os.path.join(pred_dir, f"scr2_{fname}"))
        elif lower.startswith("advanced_analysis") and lower.endswith(".txt"):
            shutil.move(fpath, os.path.join(pred_dir, f"scr2_{fname}"))

# Main execution
def main():
    organize_old_scr2_files()
    print("🧹 SCR2: using predictions\\deepseek_scr2 for all outputs...")
    
    print("=== ULTIMATE NUMBER PREDICTOR ===")
    print("🚀 Using Advanced Pattern Recognition + Ensemble Methods")
    
    predictor = UltimateNumberPredictor()
    file_path = 'number prediction learn.xlsx'
    
    # Load data
    df = predictor.load_data(file_path)
    if df is None or len(df) == 0:
        print("Trying alternative data loading method...")
        df = load_data_alternative(file_path)

    if df is not None and len(df) > 0:
        df = normalize_date_column(df)
        print("✅ Data loaded successfully!")
        df_long = predictor._ensure_long_format(df)
        print(f"📊 Total records: {len(df_long)}")

        # Keep analysis and modeling consistent by using long-format data
        
        # Enhanced analysis
        try:
            enhanced_analysis(predictor, df_long)
        except Exception as e:
            print(f"⚠️  Enhanced analysis skipped: {e}")

        # Generate predictions
        print("\n🎯 Generating advanced predictions...")
        predictions = predictor.generate_predictions(df_long, days=3, top_k=5)
        
        # Generate bet plan
        bet_plan = predictor.generate_bet_plan(predictions)
        
        # Create output files with new paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        predictions_path = os.path.join(predictor.predictions_dir, f"scr2_predictions_{timestamp}.xlsx")
        detailed_path = os.path.join(predictor.predictions_dir, f"scr2_detailed_predictions_{timestamp}.xlsx")
        bet_plan_path = os.path.join(predictor.predictions_dir, f"scr2_bet_plan_{timestamp}.xlsx")
        
        wide_predictions = predictor.create_prediction_sheet(predictions, 'wide')
        wide_predictions.to_excel(predictions_path, index=False)
        predictions.to_excel(detailed_path, index=False)
        bet_plan.to_excel(bet_plan_path, index=False)
        
        print("✅ SCR2 predictions generated successfully!")
        print("💾 Files saved:")
        print(f"   - {os.path.relpath(predictions_path, predictor.base_dir)}")
        print(f"   - {os.path.relpath(detailed_path, predictor.base_dir)}")
        print(f"   - {os.path.relpath(bet_plan_path, predictor.base_dir)}")
        
        # Display predictions
        if len(wide_predictions) > 0:
            first_date_raw = wide_predictions['date'].iloc[0]
            # Convert to a proper date object
            first_date = pd.to_datetime(first_date_raw).date()
            today = datetime.now().date()

            if first_date == today + timedelta(days=1):
                print(f"\n🎲 TOP PREDICTIONS FOR TOMORROW ({first_date.strftime('%Y-%m-%d')}):")
            else:
                print(f"\n🎲 PREDICTIONS FOR NEXT UNKNOWN RESULT DATE: {first_date.strftime('%Y-%m-%d')}")

            first_date_str = first_date.strftime('%Y-%m-%d')
            tomorrow_pred = wide_predictions[wide_predictions['date'] == first_date_str].iloc[0]
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in tomorrow_pred:
                    print(f"   {slot_name}: {tomorrow_pred[slot_name]}")

            # Display ANDAR/BAHAR
            print("\n🎯 ANDAR/BAHAR DIGITS:")
            tomorrow_bet = bet_plan[bet_plan['date'] == first_date_str]
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                slot_bet = tomorrow_bet[tomorrow_bet['slot_name'] == slot_name]
                if len(slot_bet) > 0:
                    andar = slot_bet['andar_digit'].iloc[0]
                    bahar = slot_bet['bahar_digit'].iloc[0]
                    print(f"   {slot_name}: ANDAR={andar}, BAHAR={bahar}")
        
    else:
        print("❌ Failed to load data.")

if __name__ == "__main__":
    main()