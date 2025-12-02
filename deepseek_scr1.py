import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
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

class PreciseNumberPredictor:
    def __init__(self):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}

    def _slot_name_to_id(self):
        return {name: sid for sid, name in self.slot_names.items()}

    def ensure_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the dataframe has ['date', 'slot', 'number'] columns.

        The Excel source often comes in wide form with DATE + slot columns (FRBD,
        GZBD, GALI, DSWR). This helper normalizes it to long format with numeric
        slot ids so downstream logic can work consistently.
        """

        df = df.copy()

        # Normalize date column
        if 'date' not in df.columns and 'DATE' in df.columns:
            df['date'] = pd.to_datetime(df['DATE'], errors='coerce')

        slot_map = self._slot_name_to_id()

        # If already long, just clean types
        if 'slot' in df.columns and 'number' in df.columns:
            df['slot'] = df['slot'].apply(lambda x: slot_map.get(x, x))
            df['number'] = pd.to_numeric(df['number'], errors='coerce')
            df = df.dropna(subset=['date', 'slot', 'number'])
            df['slot'] = df['slot'].astype(int)
            df['number'] = df['number'].astype(int) % 100
            return df[['date', 'slot', 'number']]

        # Otherwise convert from wide
        slot_cols = [c for c in ['FRBD', 'GZBD', 'GALI', 'DSWR'] if c in df.columns]
        if not slot_cols:
            raise ValueError("No slot columns found (expected FRBD, GZBD, GALI, DSWR).")

        parts = []
        for col in slot_cols:
            part = df[['date', col]].copy()
            part = part.rename(columns={col: 'number'})
            part['slot'] = slot_map.get(col, col)
            parts.append(part)

        long_df = pd.concat(parts, ignore_index=True)
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
            # Remove any non-digit characters except minus sign
            digits = ''.join([c for c in s if c.isdigit()])
            if not digits:
                return None
            
            # Convert to integer and take modulo 100
            num = int(digits)
            return num % 100
        except Exception as e:
            print(f"Error cleaning number {x}: {e}")
            return None

    def _two_digit(self, n:int) -> str:
        return f"{int(n)%100:02d}"

    def _ewma_weights(self, n:int, halflife:float=30.0):
        import numpy as np
        if n <= 0:
            return []
        idx = np.arange(n)
        w = 0.5 ** ((n-1 - idx)/halflife)
        s = w.sum()
        return (w / s) if s > 0 else np.ones(n)/n

    def _normalize(self, scores:dict):
        import numpy as np
        if not scores:
            return {}
        arr = np.array(list(scores.values()), dtype=float)
        if np.all(arr == 0):
            return {k:0.0 for k in scores}
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-12:
            return {k:0.0 for k in scores}
        return {k:(float(v)-mn)/(mx-mn) for k,v in scores.items()}

    def _build_components(self, df, halflife_slot:float=30.0):
        """
        Build per-slot recency, transitions (lag-1), day-of-week bonus.
        df has columns: date (datetime), slot (1..4), number (0..99 int)
        """
        import pandas as pd
        comps = {'rec_slot':{}, 'last_seen':{}, 'trans_prob':{}, 'dow_bonus':{}}

        # per-slot recency & last seen & transitions
        for s in [1,2,3,4]:
            sub = df[df['slot']==s].sort_values('date').reset_index(drop=True)
            if len(sub)==0:
                comps['rec_slot'][s] = {n:0.0 for n in range(100)}
                continue
            ws = self._ewma_weights(len(sub), halflife=halflife_slot)
            sub['w'] = ws
            rec = {n: float(sub.loc[sub['number']==n, 'w'].sum()) for n in range(100)}
            comps['rec_slot'][s] = self._normalize(rec)
            comps['last_seen'][s] = int(sub.loc[len(sub)-1, 'number'])

            # transitions with small smoothing
            counts = {i:{j:1e-3 for j in range(100)} for i in range(100)}
            for k in range(1, len(sub)):
                prev_n = int(sub.loc[k-1,'number'])
                curr_n = int(sub.loc[k,  'number'])
                counts[prev_n][curr_n] += 1.0
            trans = {}
            for i in range(100):
                row = counts[i]
                ssum = sum(row.values())
                trans[i] = {j: (row[j]/ssum) for j in range(100)} if ssum>0 else {j:0.0 for j in range(100)}
            comps['trans_prob'][s] = trans

        # DOW bonus
        df = df.copy()
        df['dow'] = df['date'].dt.dayofweek
        for d in range(7):
            sub = df[df['dow']==d]
            if len(sub)==0:
                comps['dow_bonus'][d] = {n:0.0 for n in range(100)}
                continue
            vc = sub['number'].value_counts()
            mx = vc.max()
            comps['dow_bonus'][d] = {int(n):(c/mx) for n,c in vc.items()}
            for n in range(100):
                comps['dow_bonus'][d].setdefault(n, 0.0)
        return comps

    def _scores_for_slot(self, comps, target_dow:int, slot:int, weights=(0.5,0.35,0.15)):
        a,b,c = weights
        rec = comps['rec_slot'].get(slot, {n:0.0 for n in range(100)})
        lastn = comps['last_seen'].get(slot, None)
        if lastn is None:
            trans = {n:0.0 for n in range(100)}
        else:
            trans_row = comps['trans_prob'][slot][lastn]
            trans = self._normalize(trans_row)
        dow_s = comps['dow_bonus'].get(target_dow, {n:0.0 for n in range(100)})

        scores = {}
        for n in range(100):
            scores[n] = a*rec.get(n,0.0) + b*trans.get(n,0.0) + c*dow_s.get(n,0.0)
        return scores

    def generate_predictions(self, df, days:int=3, top_k:int=5, halflife_slot:float=30.0, weights=(0.5,0.35,0.15)):
        """
        Returns a DataFrame with columns: date (str YYYY-MM-DD), slot, rank, number, score
        Start from the day AFTER the latest date present in df.
        """
        import pandas as pd
        from datetime import timedelta

        df = df.copy().sort_values(['date','slot'])
        start_date = (df['date'].max() + timedelta(days=1)).date()
        target_dates = [start_date + timedelta(days=i) for i in range(days)]

        comps = self._build_components(df, halflife_slot=halflife_slot)

        rows = []
        for d in target_dates:
            dow = pd.Timestamp(d).dayofweek
            for s in [1,2,3,4]:
                sc = self._scores_for_slot(comps, dow, s, weights)
                picks = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:top_k]
                for rank,(num,score) in enumerate(picks, start=1):
                    rows.append({
                        'date': pd.Timestamp(d).strftime('%Y-%m-%d'),
                        'slot': s,
                        'rank': rank,
                        'number': self._two_digit(num),
                        'score': float(score)
                    })
        import pandas as pd
        return pd.DataFrame(rows, columns=['date','slot','rank','number','score'])

    def create_prediction_sheet(self, predictions_df, mode='wide'):
        """
        Wide format: rows=dates; columns=FRBD,GZBD,GALI,DSWR; cells='n1, n2, n3, ...'
        """
        import pandas as pd
        pred = predictions_df.copy()
        pred['date'] = pd.to_datetime(pred['date'])
        pred = pred.sort_values(['date','slot','rank'])

        # make comma-joined per (date, slot)
        pred['num_str'] = pred.groupby(['date','slot'])['number'].transform(lambda s: ", ".join(s.astype(str)))
        uniq = pred.drop_duplicates(['date','slot'])[['date','slot','num_str']]

        wide = uniq.pivot(index='date', columns='slot', values='num_str').reset_index()
        # rename columns to slot names
        cols = ['date']
        for c in wide.columns[1:]:
            cols.append(self.slot_names.get(int(c), f"Slot{int(c)}"))
        wide.columns = cols
        wide['date'] = wide['date'].dt.date.astype(str)
        return wide

# Alternative data loading method for your specific structure
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

# Enhanced Features (Optional - Uncomment if you want to use these)
def enhance_with_advanced_features(predictor, df):
    """Add advanced features to the predictor"""
    print("\n=== Advanced Feature Analysis ===")
    
    # Prime Number Analysis
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    # Sequence Detection
    def detect_sequences(numbers, min_length=3):
        sequences = []
        current_seq = [numbers[0]]
        
        for i in range(1, len(numbers)):
            diff = numbers[i] - numbers[i-1]
            if abs(diff) <= 5:  # Small difference, likely sequence
                current_seq.append(numbers[i])
            else:
                if len(current_seq) >= min_length:
                    sequences.append(current_seq)
                current_seq = [numbers[i]]
        
        if len(current_seq) >= min_length:
            sequences.append(current_seq)
        return sequences
    
    # Apply enhancements to each slot
    for slot in [1, 2, 3, 4]:
        slot_data = df[df['slot'] == slot]
        numbers = slot_data['number'].tolist()
        
        if not numbers:
            continue
            
        # Prime number analysis
        primes = [n for n in numbers if is_prime(n)]
        prime_ratio = len(primes) / len(numbers)
        
        # Even/Odd analysis
        evens = [n for n in numbers if n % 2 == 0]
        even_ratio = len(evens) / len(numbers)
        
        print(f"Slot {predictor.slot_names[slot]}:")
        print(f"  Prime numbers: {len(primes)} ({prime_ratio:.1%})")
        print(f"  Even numbers: {len(evens)} ({even_ratio:.1%})")
        
        # Detect sequences
        sequences = detect_sequences(numbers)
        if sequences:
            print(f"  Found {len(sequences)} number sequences:")
            for seq in sequences[:2]:  # Show first 2 sequences
                print(f"    {seq}")

# Main execution function
def main():
    print("=== Precise Number Predictor ===")
    
    # Initialize predictor
    predictor = PreciseNumberPredictor()
    
    # Load your data - UPDATE THIS PATH TO YOUR ACTUAL FILE
    file_path = 'number prediction learn.xlsx'  # Change this to your file path
    
    # Try the main method first
    df = predictor.load_data(file_path)

    # If that fails, try alternative method
    if df is None or len(df) == 0:
        print("Trying alternative data loading method...")
        df = load_data_alternative(file_path)

    if df is not None and len(df) > 0:
        df = normalize_date_column(df)
        df_long = predictor.ensure_long_format(df)
        print("✅ Data loaded successfully!")
        print(f"📊 Total records: {len(df_long)}")
        print(f"📅 Date range: {df_long['date'].min().strftime('%Y-%m-%d')} to {df_long['date'].max().strftime('%Y-%m-%d')}")
        
        # Show data summary by slot
        print("\n📈 Data summary by slot:")
        for slot in [1, 2, 3, 4]:
            slot_data = df_long[df_long['slot'] == slot]
            if len(slot_data) > 0:
                print(f"  {predictor.slot_names[slot]}: {len(slot_data)} records, numbers: {slot_data['number'].min():02d}-{slot_data['number'].max():02d}")
        
        # Run advanced features analysis (optional)
        try:
            enhance_with_advanced_features(predictor, df_long)
        except Exception as e:
            print(f"⚠️  Advanced features skipped: {e}")
        
        # Generate predictions
        print("\n🎯 Generating predictions...")
        predictions = predictor.generate_predictions(df_long, days=3, top_k=5)
        
        # Create wide format sheet
        wide_predictions = predictor.create_prediction_sheet(predictions, 'wide')
        
        # Save results to root (for backward compatibility)
        root_precise_path = 'precise_predictions.xlsx'
        root_detailed_path = 'detailed_predictions.xlsx'
        
        wide_predictions.to_excel(root_precise_path, index=False)
        predictions.to_excel(root_detailed_path, index=False)
        
        # Create organized folder structure
        base_dir = os.path.dirname(os.path.abspath(__file__))
        scr1_out_dir = os.path.join(base_dir, "predictions", "deepseek_scr1")
        os.makedirs(scr1_out_dir, exist_ok=True)
        
        # Create timestamped copies
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scr1_precise_hist = os.path.join(scr1_out_dir, f"scr1_precise_predictions_{ts}.xlsx")
        scr1_detailed_hist = os.path.join(scr1_out_dir, f"scr1_detailed_predictions_{ts}.xlsx")
        
        shutil.copy2(root_precise_path, scr1_precise_hist)
        shutil.copy2(root_detailed_path, scr1_detailed_hist)
        
        print("✅ SCR1 baseline predictions saved.")
        print("   Root files (used by SCR9):")
        print("     - precise_predictions.xlsx")
        print("     - detailed_predictions.xlsx")
        print("   History copies:")
        print(f"     - predictions\\deepseek_scr1\\scr1_precise_predictions_{ts}.xlsx")
        print(f"     - predictions\\deepseek_scr1\\scr1_detailed_predictions_{ts}.xlsx")
        
        # Display tomorrow's predictions
        if len(wide_predictions) > 0:
            print("\n🎲 Top predictions for tomorrow:")
            tomorrow_pred = wide_predictions.iloc[0]
            for slot_name in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot_name in tomorrow_pred:
                    print(f"   {slot_name}: {tomorrow_pred[slot_name]}")
        
    else:
        print("❌ Failed to load data. Please check the file path and structure.")

# Run the script
if __name__ == "__main__":
    main()