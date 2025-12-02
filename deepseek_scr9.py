# deepseek_scr9.py - MODIFIED WITH SPEED MODE
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
import os
import glob
import subprocess
import re
import time
import json
import quant_data_core
warnings.filterwarnings('ignore')

class UltimatePredictionEngine:
    """
    ULTIMATE PREDICTION ENGINE - WITH SPEED MODE
    ✓ Full mode: All scripts (24min) 
    ✓ Fast mode: Critical scripts only (10-12min)
    """
    
    def __init__(self, speed_mode='full'):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.speed_mode = speed_mode  # 'full' or 'fast'
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_weights = self.load_script_weights()
        # Cache to avoid rerunning heavy scripts (e.g., SCR6) for the same date/mode within a run
        self.script_prediction_cache = {}
        self.setup_directories()

    def setup_directories(self):
        """Create output folders"""
        base_dir = self.base_dir
        folders = [
            'outputs/predictions', 
            'outputs/analysis', 
            'logs/performance', 
            'logs/prediction_results',
            'predictions/deepseek_scr9',
            'predictions/deepseek_scr1',
            'predictions/deepseek_scr2', 
            'predictions/deepseek_scr3',
            'predictions/deepseek_scr4',
            'predictions/deepseek_scr5',
            'predictions/deepseek_scr6',
            'predictions/deepseek_scr7',
            'predictions/deepseek_scr8',
        ]
        for folder in folders:
            full_path = os.path.join(base_dir, folder)
            os.makedirs(full_path, exist_ok=True)

        self.move_old_files_to_logs()

    def load_script_weights(self):
        """Load performance-based script weights (defaults to 1.0)."""
        weight_files = [
            os.path.join(self.base_dir, "logs", "performance", "script_weights.json"),
            os.path.join(self.base_dir, "logs", "performance", "script_performance_summary.json"),
        ]
        weights = {}
        loaded = False

        for path in weight_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                script_rows = data.get("scripts") if isinstance(data, dict) else None
                if isinstance(data, dict) and not script_rows:
                    # Allow direct mapping of name → weight
                    weights.update({str(k).upper(): float(v) for k, v in data.items() if v is not None})
                    loaded = True
                    break
                for row in script_rows or []:
                    name = str(row.get("name", "")).upper()
                    weight_val = row.get("weight", 1.0)
                    if not weight_val:
                        weight_val = 1.0
                    weights[name] = weight_val
                if weights:
                    loaded = True
                    break
            except Exception:
                continue

        if not loaded:
            print("⚠️  Script weight file missing; using uniform weights.")
        return weights

    def _script_key(self, script_name: str) -> str:
        match = re.search(r"scr(\d+)", script_name, re.IGNORECASE)
        if match:
            return f"SCR{int(match.group(1))}"
        return script_name.upper()
    
    def move_old_files_to_logs(self):
        """Move old prediction files to organized folders"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        movement_rules = [
            (['ultimate_*.xlsx', 'ultimate_*.txt', 'ultimate_*.csv'], 'logs/prediction_results'),
            (['ultimate_predictions.xlsx', 'ultimate_detailed_predictions.xlsx', 'bet_plan.xlsx'], 'predictions/deepseek_scr2'),
            (['advanced_analysis_*.txt', 'advanced_detailed_*.xlsx', 'prediction_diagnostic*.xlsx'], 'predictions/deepseek_scr7'),
            (['scr10_predictions_*.xlsx', 'scr10_detailed_*.xlsx', 'scr10_analysis_*.txt', 'scr10_diagnostic*.xlsx'], 'predictions/deepseek_scr8'),
            (['scr10_performance*.csv'], 'logs/performance'),
            (['ultimate_performance*.csv'], 'logs/performance'),
        ]
        
        print("🧹 Organizing files into structured folders...")
        
        for patterns, dest_folder in movement_rules:
            dest_dir = os.path.join(base_dir, dest_folder)
            os.makedirs(dest_dir, exist_ok=True)
            
            for pattern in patterns:
                for file_path in glob.glob(os.path.join(base_dir, pattern)):
                    if os.path.isfile(file_path):
                        try:
                            filename = os.path.basename(file_path)
                            dest_path = os.path.join(dest_dir, filename)
                            
                            if os.path.exists(dest_path):
                                name, ext = os.path.splitext(filename)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                dest_path = os.path.join(dest_dir, f"{name}_{timestamp}{ext}")
                            
                            os.rename(file_path, dest_path)
                            print(f"📁 Moved: {filename} -> {dest_folder}/")
                        except Exception as e:
                            print(f"⚠️ Could not move {file_path}: {e}")
    
    def load_data(self, file_path):
        """Load Excel data using shared quant_data_core loader"""
        print("📂 Loading Excel file via quant_data_core...")
        try:
            df = quant_data_core.load_results_dataframe()
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return None

        required_cols = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Results DataFrame missing required columns: {missing}")
            return None

        slot_mapping = [
            ("FRBD", 1),
            ("GZBD", 2),
            ("GALI", 3),
            ("DSWR", 4),
        ]

        all_data = []

        for _, row in df.iterrows():
            date_val = row["DATE"]
            if pd.isna(date_val):
                continue

            try:
                date_val = pd.to_datetime(date_val)
            except Exception:
                continue

            for col_name, slot_idx in slot_mapping:
                raw_val = row.get(col_name)
                if pd.isna(raw_val):
                    continue

                s = str(raw_val).strip()
                if not s or s.upper() == "XX":
                    continue

                try:
                    num = int(float(s)) % 100
                except Exception:
                    continue

                all_data.append({
                    "date": date_val,
                    "slot": slot_idx,
                    "number": num,
                })

        df_clean = pd.DataFrame(all_data)
        if df_clean.empty:
            print("❌ No valid data found after parsing results DataFrame")
            return None

        df_clean["date"] = pd.to_datetime(df_clean["date"])
        df_clean = df_clean.sort_values(["date", "slot"]).reset_index(drop=True)

        start_date = df_clean["date"].min().strftime("%Y-%m-%d")
        end_date = df_clean["date"].max().strftime("%Y-%m-%d")
        print(f"✅ Loaded {len(df_clean)} records from {start_date} to {end_date}")

        return df_clean
    
    def get_opposite(self, n):
        """Get opposite number"""
        if n < 10:
            return n * 10
        else:
            return (n % 10) * 10 + (n // 10)
    
    def run_child_script(self, script_name, target_date=None, mode=None):
        """Run a child script and parse its predictions with optional caching"""
        try:
            cache_key = None
            timeout_seconds = 300

            if script_name.lower() == 'deepseek_scr6.py':
                # Heavier model ensemble – cache per target_date + mode within the same run
                cache_key = (script_name, str(target_date) if target_date else None, mode or 'default')
                if cache_key in self.script_prediction_cache:
                    print(
                        "   🔁 Using cached predictions for deepseek_scr6.py "
                        f"({cache_key[1]}, mode={cache_key[2]})"
                    )
                    cached_preds = self.script_prediction_cache[cache_key]
                    return {slot: list(nums) for slot, nums in cached_preds.items()}
                timeout_seconds = None  # Allow generous runtime for SCR6 without killing the process

            print(f"   🔄 Running {script_name}...")

            result = subprocess.run(
                ['py', '-3.12', script_name],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            
            predictions = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
            output_lines = result.stdout.split('\n')
            
            for line in output_lines:
                line = line.strip()
                
                for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
                    if slot in line and (':' in line or '(' in line):
                        if ':' in line:
                            numbers_part = line.split(':', 1)[1]
                        elif '(' in line:
                            numbers_part = line.split(')', 1)[1]
                        else:
                            continue
                            
                        numbers = re.findall(r'\b\d{2}\b', numbers_part)
                        for num_str in numbers:
                            try:
                                num = int(num_str)
                                if 0 <= num <= 99 and num not in predictions[slot]:
                                    predictions[slot].append(num)
                            except:
                                continue
                
                if "Hot numbers:" in line:
                    hot_numbers = re.findall(r"'(\d+)", line)
                    for num_str in hot_numbers:
                        try:
                            num = int(num_str) % 100
                            for slot in predictions:
                                if num not in predictions[slot]:
                                    predictions[slot].append(num)
                        except:
                            continue
            
            if cache_key:
                # Store a shallow copy to protect cache from accidental mutation
                cached_copy = {slot: list(nums) for slot, nums in predictions.items()}
                self.script_prediction_cache[cache_key] = cached_copy

            return predictions

        except subprocess.TimeoutExpired:
            print(f"   ⚠️  {script_name} timed out after {timeout_seconds} seconds")
            return {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        except Exception as e:
            print(f"   ⚠️  {script_name} failed: {e}")
            return {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}

    def collect_all_script_predictions(self, target_date=None, mode=None):
        """Collect predictions from all available scripts - OPTIMIZED FOR SPEED MODE"""
        print("🎯 Collecting predictions from scripts...")
        
        if self.speed_mode == 'fast':
            print("   ⚡ FAST MODE: Running SCR1–SCR8 once for tomorrow only")
        else:
            print("   🐢 FULL MODE: Running all scripts")

        scripts = [
            'deepseek_scr1.py',
            'deepseek_scr2.py',
            'deepseek_scr3.py',
            'deepseek_scr4.py',
            'deepseek_scr5.py',
            'deepseek_scr6.py',
            'deepseek_scr7.py',
            'deepseek_scr8.py',
        ]
        
        all_predictions = {}
        script_times = {}
        
        for script in scripts:
            if os.path.exists(script):
                start_time = time.time()
                preds = self.run_child_script(script, target_date=target_date, mode=mode)
                end_time = time.time()
                script_times[script] = end_time - start_time
                all_predictions[script] = preds
            else:
                print(f"   ⚠️  {script} not found")
        
        # Print timing summary
        if script_times:
            print("   ⏰ Script execution times:")
            for script, exec_time in script_times.items():
                print(f"     {script}: {exec_time:.1f}s")
        
        return all_predictions
    
    def build_slot_scores(self, all_preds_for_slot, history_df_for_slot):
        """Build ensemble scores for a slot - OPTIMIZED"""
        scores = Counter()
        
        # Process each script's predictions
        for script_name, predictions in all_preds_for_slot.items():
            weight = self.script_weights.get(self._script_key(script_name), 1.0)

            for rank, number in enumerate(predictions, 1):
                if rank == 1:
                    rank_weight = 5
                elif rank == 2:
                    rank_weight = 4
                elif rank == 3:
                    rank_weight = 3
                elif rank == 4:
                    rank_weight = 2
                else:
                    rank_weight = 1
                
                scores[number] += rank_weight * weight
        
        # Frequency bonus (optimized for speed)
        frequency_bonus = Counter()
        for script_name, predictions in all_preds_for_slot.items():
            for number in predictions[:8]:  # Reduced from 10 to 8 in fast mode
                frequency_bonus[number] += 1
        
        for number, freq in frequency_bonus.items():
            if freq >= 2:  # Reduced threshold in fast mode
                scores[number] += freq * 2
        
        # Opposite number bonus
        high_score_numbers = [num for num, score in scores.most_common(8)]  # Reduced from 10
        for number in high_score_numbers:
            opposite = self.get_opposite(number)
            scores[opposite] += 3
        
        # Recent actual results bonus
        if len(history_df_for_slot) > 0:
            recent_numbers = history_df_for_slot['number'].tail(3).tolist()
            for number in recent_numbers:
                scores[number] += 2
                scores[self.get_opposite(number)] += 1
        
        return scores
    
    def predict_for_target_date(self, df_history, target_date):
        """Generate predictions for a specific target date - OPTIMIZED"""
        print(f"   🎯 Predicting for {target_date}...")
        
        # Collect predictions from all scripts
        all_script_preds = self.collect_all_script_predictions(target_date=target_date, mode='predict_for_date')
        
        # Reorganize by slot
        slot_predictions = {}
        for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
            slot_preds = {}
            for script_name, predictions in all_script_preds.items():
                slot_preds[script_name] = predictions[slot]
            slot_predictions[slot] = slot_preds
        
        # Build final predictions for each slot
        final_predictions = {}
        
        for slot_name in ["FRBD", "GZBD", "GALI", "DSWR"]:
            slot_data = df_history[df_history['slot'] == list(self.slot_names.keys())[list(self.slot_names.values()).index(slot_name)]]
            numbers = slot_data['number'].tolist()
            
            if len(numbers) < 5:
                # Fallback: simple frequency
                freq = Counter(numbers)
                final_predictions[slot_name] = [num for num, count in freq.most_common(15)]
            else:
                # Use ensemble scoring
                scores = self.build_slot_scores(slot_predictions[slot_name], slot_data)
                
                # Dynamic top-k selection (optimized)
                if scores:
                    top_scores = [score for num, score in scores.most_common(8)]  # Reduced from 10
                    if len(top_scores) > 1:
                        score_ratio = top_scores[0] / top_scores[1] if top_scores[1] > 0 else 10
                        if score_ratio > 2.0:
                            top_k = 5
                        elif score_ratio > 1.5:
                            top_k = 10
                        else:
                            top_k = 15
                    else:
                        top_k = 15
                else:
                    top_k = 15
                
                # Apply range diversity
                candidates = [num for num, score in scores.most_common(top_k * 2)]
                final_pred = self.apply_diversity_filter({num: 1.0 for num in candidates}, top_k)
                final_predictions[slot_name] = final_pred
        
        return final_predictions
    
    def apply_diversity_filter(self, scores, top_k):
        """Ensure diversity across ranges"""
        if not scores:
            return list(range(top_k))
        
        sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [num for num, _ in sorted_preds[:top_k * 2]]
        
        # Group by ranges
        ranges = {
            'low': [n for n in candidates if 0 <= n <= 33],
            'medium': [n for n in candidates if 34 <= n <= 66],
            'high': [n for n in candidates if 67 <= n <= 99]
        }
        
        selected = []
        
        # Take top from each range
        for rng in ['low', 'medium', 'high']:
            if ranges[rng]:
                selected.append(ranges[rng][0])
        
        # Fill remaining
        for num in candidates:
            if num not in selected and len(selected) < top_k:
                selected.append(num)
        
        return selected[:top_k]
    
    def backtest_recent_days(self, df, days=3):
        """Backtest on recent complete days - OPTIMIZED FOR SPEED MODE"""
        print(f"\n📊 Running backtest for last {days} days...")
        
        # SPEED MODE OPTIMIZATION: Reduce backtest days
        if self.speed_mode == 'fast':
            days = 1  # Only test most recent day in fast mode
            print("   🚀 SPEED MODE: Testing only most recent day")
        
        try:
            # Find dates with all four slots
            date_counts = df.groupby('date')['slot'].nunique()
            complete_dates = date_counts[date_counts == 4].index.tolist()
            complete_dates.sort()
            
            if len(complete_dates) < days:
                print(f"⚠️  Only {len(complete_dates)} complete dates found, using all")
                test_dates = complete_dates[-days:] if len(complete_dates) > 0 else []
            else:
                test_dates = complete_dates[-days:]
            
            if not test_dates:
                print("❌ No complete dates found for backtesting")
                return pd.DataFrame()
            
            backtest_results = []
            
            for test_date in test_dates:
                test_date_str = test_date.strftime('%Y-%m-%d')
                print(f"   🔍 Testing {test_date_str}...")
                
                # Split data
                history_df = df[df['date'] < test_date]
                actual_df = df[df['date'] == test_date]
                
                if len(history_df) == 0:
                    continue
                
                # Get predictions for this test date
                predictions = self.predict_for_target_date(history_df, test_date)
                
                # Compare with actuals
                for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
                    actual_data = actual_df[actual_df['slot'] == list(self.slot_names.keys())[list(self.slot_names.values()).index(slot)]]
                    if not actual_data.empty:
                        actual_number = actual_data['number'].iloc[0]
                        pred_numbers = predictions.get(slot, [])
                        
                        # Find rank of actual number
                        rank = None
                        if actual_number in pred_numbers:
                            rank = pred_numbers.index(actual_number) + 1
                        
                        # Check hits
                        hit_top1 = rank == 1 if rank else False
                        hit_top5 = rank <= 5 if rank else False
                        hit_top10 = rank <= 10 if rank else False
                        hit_top15 = rank <= 15 if rank else False
                        
                        backtest_results.append({
                            'date': test_date_str,
                            'slot': slot,
                            'actual': actual_number,
                            'predictions': ','.join(map(str, pred_numbers)),
                            'rank': rank,
                            'hit_top1': hit_top1,
                            'hit_top5': hit_top5,
                            'hit_top10': hit_top10,
                            'hit_top15': hit_top15
                        })
            
            results_df = pd.DataFrame(backtest_results)
            
            # Save to performance log
            if not results_df.empty:
                perf_file = 'logs/performance/ultimate_performance.csv'
                if os.path.exists(perf_file):
                    existing_df = pd.read_csv(perf_file)
                    combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date', 'slot'], keep='last')
                    combined_df.to_csv(perf_file, index=False)
                else:
                    results_df.to_csv(perf_file, index=False)
                
                print(f"✅ Backtest results saved to {perf_file}")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return pd.DataFrame()
    
    def detect_status(self, df):
        """Detect current date status"""
        latest_date = df['date'].max()
        today_data = df[df['date'] == latest_date]
        filled_slots = set(today_data['slot'].tolist())
        empty_slots = [s for s in [1, 2, 3, 4] if s not in filled_slots]
        return latest_date, filled_slots, empty_slots
    
    def generate_predictions(self, df):
        """Generate complete predictions"""
        latest_date, filled_slots, empty_slots = self.detect_status(df)
        
        print(f"\n📅 Latest Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"✅ Filled: {[self.slot_names[s] for s in filled_slots]}")
        print(f"❌ Empty: {[self.slot_names[s] for s in empty_slots]}")
        
        predictions = []
        
        # TODAY'S EMPTY SLOTS (skip in FAST mode)
        if self.speed_mode != 'fast' and empty_slots:
            print(f"\n🎯 Predicting TODAY's empty slots...")
            date_str = latest_date.strftime('%Y-%m-%d')

            for slot in empty_slots:
                slot_name = self.slot_names[slot]
                slot_data = df[df['slot'] == slot]
                numbers = slot_data['number'].tolist()

                # Use ensemble prediction
                all_script_preds = self.collect_all_script_predictions(target_date=latest_date, mode='today_empty')
                slot_preds = {}
                for script_name, preds in all_script_preds.items():
                    slot_preds[script_name] = preds[slot_name]

                scores = self.build_slot_scores(slot_preds, slot_data)
                pred_nums = [num for num, score in scores.most_common(15)]
                opposites = [self.get_opposite(n) for n in pred_nums[:3]]

                for rank, num in enumerate(pred_nums, 1):
                    predictions.append({
                        'date': date_str,
                        'slot': slot_name,
                        'number': f"{num:02d}",
                        'rank': rank,
                        'type': 'TODAY_EMPTY',
                        'opposites': ', '.join([f"{n:02d}" for n in opposites]) if rank == 1 else ''
                    })
        elif self.speed_mode == 'fast' and empty_slots:
            print("\n⚡ FAST MODE: Skipping today's empty-slot predictions")
        
        # TOMORROW'S ALL SLOTS
        tomorrow = latest_date + timedelta(days=1)
        date_str = tomorrow.strftime('%Y-%m-%d')
        print(f"\n🎯 Predicting TOMORROW ({date_str})...")
        
        # Use the main prediction function
        tomorrow_preds = self.predict_for_target_date(df, tomorrow)
        
        for slot_name, pred_numbers in tomorrow_preds.items():
            opposites = [self.get_opposite(n) for n in pred_numbers[:3]]
            
            for rank, num in enumerate(pred_numbers, 1):
                predictions.append({
                    'date': date_str,
                    'slot': slot_name,
                    'number': f"{num:02d}",
                    'rank': rank,
                    'type': 'TOMORROW',
                    'opposites': ', '.join([f"{n:02d}" for n in opposites]) if rank == 1 else ''
                })
        
        return pd.DataFrame(predictions)
    
    def create_outputs(self, predictions_df, df, backtest_results):
        """Create output files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        scr9_pred_dir = os.path.join(base_dir, "predictions", "deepseek_scr9")
        
        # Ensure the directory exists
        os.makedirs(scr9_pred_dir, exist_ok=True)
        
        # Wide format
        wide_data = []
        for date in predictions_df['date'].unique():
            date_data = {
                'date': date,
                'type': predictions_df[predictions_df['date'] == date]['type'].iloc[0]
            }
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                slot_data = predictions_df[
                    (predictions_df['date'] == date) & 
                    (predictions_df['slot'] == slot)
                ]
                if not slot_data.empty:
                    nums = slot_data['number'].tolist()[:15]
                    date_data[slot] = ', '.join(nums)
                    opp = slot_data['opposites'].iloc[0]
                    if pd.notna(opp) and opp:
                        date_data[f'{slot}_OPP'] = opp
            wide_data.append(date_data)
        
        wide_df = pd.DataFrame(wide_data)
        
        # Define file paths in the dedicated SCR9 folder
        pred_file = os.path.join(scr9_pred_dir, f'ultimate_predictions_{timestamp}.xlsx')
        detail_file = os.path.join(scr9_pred_dir, f'ultimate_detailed_{timestamp}.xlsx')
        diag_file = os.path.join(scr9_pred_dir, f'ultimate_diagnostic_{timestamp}.xlsx')
        analysis_file = os.path.join(scr9_pred_dir, f'ultimate_analysis_{timestamp}.txt')
        
        # Save files to dedicated folder
        wide_df.to_excel(pred_file, index=False)
        predictions_df.to_excel(detail_file, index=False)
        
        # Create diagnostic file
        diagnostic_data = []
        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_data = df[df['slot'] == list(self.slot_names.keys())[list(self.slot_names.values()).index(slot)]]
            if len(slot_data) > 0:
                recent = slot_data['number'].tail(10).tolist()
                freq = Counter(slot_data['number'])
                hot_numbers = [num for num, count in freq.most_common(5)]
                
                diagnostic_data.append({
                    'slot': slot,
                    'recent_numbers': ', '.join([f'{n:02d}' for n in recent]),
                    'hot_numbers': ', '.join([f'{n:02d}' for n in hot_numbers]),
                    'total_records': len(slot_data)
                })
        
        diag_df = pd.DataFrame(diagnostic_data)
        diag_df.to_excel(diag_file, index=False)
        
        # Report
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  🎯 SCR11 ULTIMATE PREDICTION ENGINE - REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Speed Mode: {self.speed_mode.upper()}\n")
            f.write(f"Records: {len(df)}\n")
            f.write(f"Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n\n")
            
            # Backtest summary
            if not backtest_results.empty:
                f.write("BACKTEST SUMMARY:\n")
                f.write("-" * 50 + "\n")
                for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    slot_results = backtest_results[backtest_results['slot'] == slot]
                    if not slot_results.empty:
                        total = len(slot_results)
                        hit_top5 = slot_results['hit_top5'].sum()
                        f.write(f"  {slot}: {hit_top5}/{total} hit_top5 ({hit_top5/total*100:.1f}%)\n")
                f.write("\n")
            
            # Predictions
            f.write("PREDICTIONS:\n")
            f.write("-" * 50 + "\n")
            for date in wide_df['date'].unique():
                row = wide_df[wide_df['date'] == date].iloc[0]
                f.write(f"\n{date} ({row['type']}):\n")
                for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    if slot in row:
                        f.write(f"  {slot}: {row[slot]}\n")
                        if f'{slot}_OPP' in row:
                            f.write(f"       Opp: {row[f'{slot}_OPP']}\n")

            # Script weights used
            f.write("\nSCRIPT WEIGHTS USED:\n")
            f.write("-" * 50 + "\n")
            for idx in range(1, 10):
                key = f"SCR{idx}"
                weight_val = self.script_weights.get(key, 1.0)
                f.write(f"  {key}: {weight_val:.2f}\n")

        # Return file paths for console display
        return wide_df, pred_file, detail_file, diag_file, analysis_file
    
    def run(self, file_path):
        """Main execution"""
        print("=" * 70)
        print(f"  🎯 SCR11 ULTIMATE PREDICTION ENGINE - {self.speed_mode.upper()} MODE")
        print("  ✓ TRUE SCR1-10 Integration + Backtesting + Organized Output")
        print(f"  Mode: {self.speed_mode.upper()}")
        print("=" * 70)
        
        start_time = time.time()
        
        df = self.load_data(file_path)
        if df is None:
            return
        
        # Add slot names for analysis
        df['slot_name'] = df['slot'].map(self.slot_names)
        
        print(f"\n📊 Data Summary:")
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            print(f"   {self.slot_names[slot]}: {len(slot_data)} records")
        
        # Run backtest (skip entirely in FAST mode)
        if self.speed_mode == 'fast':
            print("\n🏎 FAST MODE: Skipping 3-day backtest (using existing reality metrics)")
            backtest_results = pd.DataFrame()
        else:
            backtest_results = self.backtest_recent_days(df, days=3)

        # Print backtest summary
        if not backtest_results.empty:
            print(f"\n📊 Backtest (last {len(backtest_results['date'].unique())} days):")
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                slot_results = backtest_results[backtest_results['slot'] == slot]
                if not slot_results.empty:
                    total = len(slot_results)
                    hit_top5 = slot_results['hit_top5'].sum()
                    print(f"   {slot}: {hit_top5}/{total} hit_top5 ({hit_top5/total*100:.1f}%)")
        elif self.speed_mode == 'fast':
            print("\n📊 Backtest metrics unavailable in FAST mode (skipped to save time)")

        if self.speed_mode == 'fast':
            print("\n⚡ FAST MODE: Only generating TOMORROW predictions (no historical backtest)")
        else:
            print("\n🐢 FULL MODE: Running backtest + TOMORROW predictions")

        # Generate predictions
        predictions_df = self.generate_predictions(df)
        wide_df, pred_file, detail_file, diag_file, analysis_file = self.create_outputs(predictions_df, df, backtest_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("  📊 RESULTS")
        print("=" * 70)
        
        for date in wide_df['date'].unique():
            row = wide_df[wide_df['date'] == date].iloc[0]
            print(f"\n📅 {date} ({row['type']}):")
            print("-" * 70)
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot in row:
                    print(f"  {slot:5s}: {row[slot]}")
                    if f'{slot}_OPP' in row:
                        print(f"         Opp: {row[f'{slot}_OPP']}")
        
        print("\n" + "=" * 70)
        print("✅ Files saved:")
        print(f"   - {os.path.relpath(pred_file)}")
        print(f"   - {os.path.relpath(detail_file)}")
        print(f"   - {os.path.relpath(diag_file)}")
        print(f"   - {os.path.relpath(analysis_file)}")
        print(f"   - logs/performance/ultimate_performance.csv")
        print(f"\n⏰ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SCR11 Ultimate Prediction Engine')
    parser.add_argument('--speed-mode', choices=['full', 'fast'], default='full',
                       help='Speed mode: full (backtest + tomorrow) or fast (tomorrow only)')
    
    args = parser.parse_args()
    
    predictor = UltimatePredictionEngine(speed_mode=args.speed_mode)
    predictor.run('number prediction learn.xlsx')