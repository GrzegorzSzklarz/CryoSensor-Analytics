"""
Cryogenic Sensor Data Correction & Statistical Analysis System
==============================================================

This script provides a pipeline for correcting resistance measurements 
from various cryogenic sensors using a hybrid approach based on temperature 
ranges and sensor types.


Correction Strategies:
----------------------
1. ITS-90 (Standard): For specific Platinum sensors using ITS-90 coefficients.
2. S-Factor Integration: For low-temperature non-Pt sensors using 
   polynomial integration of sensitivity (dR/dT).
3. In-Situ Sensitivity: For high-temperature non-Pt sensors using linear 
   regression of R(T) at low excitation currents.
   
Key Features:
-------------
- Parallel Processing
- Numba Optimization: JIT-compiled mathematical kernels for ITS-90.
- Statistical Noise: Allan-style noise calculation.
- Outlier Rejection: IQR filtering for block and rolling averages.

Date: 2026
"""

import numpy as np
import pandas as pd
import os
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from numpy.lib.stride_tricks import sliding_window_view

# Suppression of RuntimeWarnings for cleaner parallel logs
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0')

# ==========================================
# --- 1. CONFIGURATION & COEFFICIENTS ---
# ==========================================

# Directory Paths (Standardized for generic repository)
DATA_DIR = '.../data/input'
COEFFS_FILE = 'coefficients.csv'
OUTPUT_DIR = '.../results/processed_data'

# Sensor categories/methods
FOLDERS_S_FACTOR = ['low_temp_range_a', 'low_temp_range_a']
FOLDERS_SENSITIVITY = ['high_temp_range_b', 'high_temp_range_b']

# Generic Sensor Identification Keys
PT_ITS90_SENSORS = ['PT_SENSOR_01', 'PT_SENSOR_02']

# Global Constants
TRIM_POINTS = 10
TARGET_CURRENTS = [0.1, 0.5, 1.0, 2.0, 5.0]

TEMPERATURE_COLUMN = 'temperature_k' 
RESISTANCE_COLUMN = 'resistance_ohm'
EXCITATION_CURRENT_COLUMN = 'excitation_current_ma'

# Sensitivity Coefficients (dR/dT) - Polynomial coefficients
# Examples kept for logic validation
S_FACTOR_COEFFS = {
    'SENSOR_TYPE_A': np.array([0.0112, -0.0068, 0.0015, -0.0001, 1e-05, -3e-07, 3e-09]),
    'SENSOR_TYPE_B': np.array([-1.3116, 0.7010, -0.1199, 0.0104, -0.0004, 1e-05, -1e-07]),
}

# ==========================================
# --- 2. Math Core ---
# ==========================================

A_COEFFS = np.array([-2.13534729, 3.18324720, -1.80143597, 0.71727204, 0.50344027, -0.61899395, -0.05332322, 0.28021362, 0.10715224, -0.29302865, 0.04459872, 0.11868632, -0.05248134])

@njit(fastmath=True)
def calc_Wr(T):
    """Calculates the ITS-90 reference function Wr(T)."""
    if T < 273.16:
        x = (np.log(max(T, 1e-10) / 273.16) + 1.5) / 1.5
        y = 0.0
        for i in range(len(A_COEFFS)-1, -1, -1):
            y += A_COEFFS[i] * (x**i)
        return np.exp(y)
    return 1.0

@njit(fastmath=True)
def solve_W(T, coeffs):
    """Calculates the deviation function and returns W(T)."""
    Wr = calc_Wr(T)
    W = Wr
    for _ in range(10):
        Wm1 = W - 1.0
        lnW = np.log(max(W, 1e-15))
        dev = coeffs[0]*Wm1 + coeffs[1]*(Wm1**2)
        p = 3
        for k in range(2, len(coeffs)):
            dev += coeffs[k] * (lnW**p)
            p += 1
        W = Wr + dev
    return W

@njit(fastmath=True)
def integrate_polynomial_s(coeffs, t_start, t_end):
    """Analytically integrates sensitivity polynomial."""
    res_start = 0.0
    res_end = 0.0
    for i in range(len(coeffs)):
        power = i + 1
        term_div = coeffs[i] / power
        res_end += term_div * (t_end**power)
        res_start += term_div * (t_start**power)
    return res_end - res_start

# ==========================================
# --- 3. STATISTICAL HELPERS ---
# ==========================================

def calc_noise_vectorized(arr):
    if len(arr) < 3: return np.nan
    diffs = np.diff(arr)
    return np.std(diffs, ddof=1) / np.sqrt(2)

def normalize_name(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def make_block_avg_simple(df, block_size):
    n = len(df)
    n_keep = n - (n % block_size)
    if n_keep == 0: return pd.DataFrame()
    df_trunc = df.iloc[:n_keep].reset_index(drop=True)
    ids = np.arange(n_keep) // block_size
    return df_trunc.groupby(ids).mean()

def make_block_avg_iqr(df, block_size):
    n = len(df)
    n_keep = n - (n % block_size)
    if n_keep == 0: return pd.DataFrame()
    df_trunc = df.iloc[:n_keep].reset_index(drop=True)
    num_cols = df_trunc.select_dtypes(include=np.number).columns
    result = {}
    for col in num_cols:
        arr = df_trunc[col].values.reshape(-1, block_size)
        q1, q3 = np.percentile(arr, [25, 75], axis=1, keepdims=True)
        iqr = q3 - q1
        mask = (arr >= q1 - 1.5*iqr) & (arr <= q3 + 1.5*iqr)
        sums = np.sum(arr * mask, axis=1)
        counts = np.sum(mask, axis=1)
        result[col] = np.divide(sums, counts, out=np.mean(arr, axis=1), where=counts!=0)
    return pd.DataFrame(result)

def make_rolling_avg(df_in, window, current_col):
    if current_col not in df_in.columns: return pd.DataFrame()
    df = df_in.copy()
    df['__block__'] = (df[current_col] != df[current_col].shift()).cumsum()
    num_cols = df.select_dtypes(include=np.number).columns
    df_rolled = df.groupby('__block__')[num_cols].rolling(window).mean()
    return df_rolled.reset_index(level=0, drop=True).sort_index()

def make_rolling_avg_iqr(df_in, window, current_col):
    if current_col not in df_in.columns: return pd.DataFrame()
    df = df_in.copy()
    df['__block__'] = (df[current_col] != df[current_col].shift()).cumsum()
    results = []
    for _, group in df.groupby('__block__'):
        if len(group) < window: continue
        group_vals = group.reset_index(drop=True)
        group_res = group_vals.iloc[window-1:].copy()
        num_cols = group.select_dtypes(include=np.number).columns
        for col in num_cols:
            vals = group_vals[col].values
            windows = sliding_window_view(vals, window_shape=window)
            q1, q3 = np.percentile(windows, [25, 75], axis=1)
            iqr_val = q3 - q1
            mask = (windows >= (q1 - 1.5 * iqr_val)[:, None]) & (windows <= (q3 + 1.5 * iqr_val)[:, None])
            group_res[col] = np.divide(np.sum(windows * mask, axis=1), np.sum(mask, axis=1), 
                                       out=np.mean(windows, axis=1), where=np.sum(mask, axis=1)!=0)
        results.append(group_res)
    return pd.concat(results).sort_index() if results else pd.DataFrame()

def get_stats_for_summary(df_in, current_col, value_col, sensor_name, t_target=np.nan):
    stats = {'Sensor': sensor_name, 'T-Target': t_target}
    if df_in.empty: return stats
    
    df = df_in.copy().dropna(subset=[value_col])
    df['__b__'] = (df[current_col] != df[current_col].shift()).cumsum()
    
    for tc in TARGET_CURRENTS:
        # Find matches for current
        blk_vals = []
        for _, blk in df.groupby('__b__'):
            if np.isclose(blk[current_col].iloc[0], tc, atol=0.01):
                if len(blk) > TRIM_POINTS:
                    blk_vals.extend(blk[value_col].iloc[TRIM_POINTS:].values)
        
        arr = np.array(blk_vals)
        avg_val = np.mean(arr) if len(arr) > 0 else np.nan
        noise = calc_noise_vectorized(arr) if len(arr) > 0 else np.nan
        
        stats[f"{tc}mA"] = avg_val
        stats[f"Szum {tc} mA"] = noise
        stats[f"SNR {tc} mA"] = 20 * np.log10(avg_val / noise) if not np.isnan(noise) and noise > 0 else np.nan
    return stats
    
# ==========================================
# --- 4. PROCESSING PIPELINE ---
# ==========================================

def process_file_pipeline(args):
    root, file, df_coeffs_dict, output_dir = args
    folder_name = os.path.basename(root)
    
    # Generic matching logic
    is_sens = any(k in folder_name for k in FOLDERS_SENSITIVITY)
    is_s_factor = not is_sens and any(k in folder_name for k in FOLDERS_S_FACTOR)
    if not (is_s_factor or is_sens): return None

    # Anonymized Sensor ID
    file_norm = normalize_name(file)
    sensor_name = next((k for k in S_FACTOR_COEFFS if normalize_name(k) in file_norm), "UNKNOWN_SENSOR")

    try:
        df = pd.read_csv(os.path.join(root, file))
        df.columns = df.columns.str.strip()
    except: return None
    
    # Use generic column names or placeholders
    t_col, r_col, c_col = TEMPERATURE_COLUMN, RESISTANCE_COLUMN, EXCITATION_CURRENT_COLUMN
    if t_col not in df.columns: return None

    time_col = next((c for c in ['Secs since 1Jan1904', 'Secs since file creation'] if c in df.columns), None)
    if not time_col: return None
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Temperature Target
    t_avg = (df[t_col].values[0] + df[t_col].values[-1]) / 2.0
    r_corr = df[r_col].values.copy()
    
    slope = 0.0
    
    # 1. ITS-90 Correction
    if sensor_name in PT_ITS90_SENSORS:
        s_simp = normalize_name(sensor_name).replace('rs','').replace('pt','')
        my_coeffs = next((v for k,v in df_coeffs_dict.items() if s_simp in normalize_name(k)), None)
        if my_coeffs is not None:
            W_targ = solve_W(t_avg, my_coeffs)
            for i in range(len(r_corr)):
                r_corr[i] *= (W_targ / solve_W(df[t_col].values[i], my_coeffs))

    # 2. S-Factor Polynomial Integration
    elif is_s_factor and sensor_name in S_FACTOR_COEFFS:
        c_poly = S_FACTOR_COEFFS[sensor_name]
        for i in range(len(r_corr)):
            r_corr[i] += integrate_polynomial_s(c_poly, df[t_col].values[i], t_avg)

    # 3. Sensitivity (Linear Regression)
    elif is_sens and c_col in df.columns:
        target_c = 5.0
        df_high = df[np.isclose(df[c_col], target_c, atol=0.01)]

        if df_high.empty:
            target_c = df[c_col].max()
            df_high = df[df[c_col] == target_c]
            
        slope = 0.0
        if len(df_high) > (2*TRIM_POINTS+5):
            df_t = df_high.iloc[TRIM_POINTS:-TRIM_POINTS]
            slope, _ = np.polyfit(df_t[t_col], df_t[r_col], 1)
            
            print(f"Sensivity: {sensor_name:<15} in {folder_name:<6} (estimated for {target_c}mA) = {slope:10.6f} Ohm/K")
            
        r_corr += slope * (t_avg - df[t_col].values)

    df['R_Corr'] = r_corr
    df_base = df[['Secs since 1Jan1904', r_col, c_col, t_col, 'R_Corr']].copy()
    
    # Generate Multi-version Output
    data_versions = {
        'Corr': df_base,
        'Avg5': make_block_avg_simple(df_base, 5),
        'Avg5Filt': make_block_avg_iqr(df_base, 5),
        'MovAvg5': make_rolling_avg(df_base, 5, c_col),
        'MovAvg5Filt': make_rolling_avg_iqr(df_base, 5, c_col),
        'Avg10': make_block_avg_simple(df_base, 10),
        'Avg10Filt': make_block_avg_iqr(df_base, 10),
        'MovAvg10': make_rolling_avg(df_base, 10, c_col),
        'MovAvg10Filt': make_rolling_avg_iqr(df_base, 10, c_col)
    }

    stats_collection = {}
    base_out_path = os.path.join(output_dir, folder_name)
    s_id = file.replace('.csv', '')
    
    for key, dframe in data_versions.items():
        sub_dir = os.path.join(base_out_path, f"data_{key}")
        os.makedirs(sub_dir, exist_ok=True)
        if not dframe.empty: dframe.to_csv(os.path.join(sub_dir, f"{key}_{file}"), index=False)
        stats_collection[key] = get_stats_for_summary(dframe, c_col, 'R_Corr', s_id, t_target=t_avg)
        
    sens_val = slope if is_sens else None
    
    return (folder_name, stats_collection, (sensor_name, sens_val))

# ==========================================
# --- 5. MAIN EXECUTION ---
# ==========================================

def main():

    print(f"--- START: Extended Stats Pipeline ({os.cpu_count()} cores) ---")
    start_t = time.time()
    
    # 1. Pre-load Coefficients (with R_tpw support)
    coeffs_dict = {}
    if os.path.exists(COEFFS_FILE):
        try:
            df_c = pd.read_csv(COEFFS_FILE)
            # We keep rows where Term_Index is not null (captures both numbers and strings like R_tpw)
            sub = df_c.dropna(subset=['Term_Index'])
            
            for col in sub.columns:
                if col not in ['Range', 'Term_Index']:
                    # pd.to_numeric handles strings and floats, dropping non-convertible garbage
                    values = pd.to_numeric(sub[col], errors='coerce').dropna().values
                    coeffs_dict[col] = values.astype(float)
            print(f"Loaded coefficients for {len(coeffs_dict)} sensors.")
        except Exception as e:
            print(f"Error loading coefficients file: {e}")
    else:
        print(f"Warning: {COEFFS_FILE} not found. Proceeding without ITS-90 correction.")

    # 2. Prepare Directory Structure
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. Walk through data directory to find measurement files
    tasks = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv'):
                tasks.append((root, f, coeffs_dict, OUTPUT_DIR))
    
    print(f"Queued {len(tasks)} files for processing...")
    
    # 4. Run Parallel Execution
    summary_storage = {}
    sens_results = [] 

    with ProcessPoolExecutor() as exe:
        # Using list() to force the iterator to complete
        results = list(exe.map(process_file_pipeline, tasks))
        
        for res in results:
            if not res: continue
            
            folder, all_stats, sens_info = res
            
            # Sensitivity estimation tracking (for high-temp sensors)
            if sens_info[1] is not None:
                sens_results.append(f"{sens_info[0]:<15} | Folder: {folder:<6} | S: {sens_info[1]:.6f} Ohm/K")
            
            # Aggregate stats for final summaries
            if folder not in summary_storage:
                summary_storage[folder] = {}
            
            for key, stats in all_stats.items():
                if key not in summary_storage[folder]:
                    summary_storage[folder][key] = []
                summary_storage[folder][key].append(stats)

# 5. Save Global Summaries per Folder
    print("Saving Folder Summaries...")
    for folder, types_dict in summary_storage.items():
        for type_key, stats_list in types_dict.items():
            if not stats_list: continue

            df_sum = pd.DataFrame(stats_list).sort_values('Sensor')
            
            # Build consistent column order
            cols = ['Sensor', 'T-Target']
            for c in TARGET_CURRENTS:
                for suffix in [f"{c}mA", f"Szum {c} mA", f"SNR {c} mA"]:
                    if suffix not in df_sum.columns:
                        df_sum[suffix] = np.nan
                    cols.append(suffix)
            
            final_path = os.path.join(OUTPUT_DIR, folder, f"SUMMARY_{type_key}_{folder}.csv")
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            df_sum[cols].to_csv(final_path, index=False)

    # =============================================================================
    # --- 6. VISUALIZATION SECTION ---
    # =============================================================================
    print("Generating Global Plots...")
    plot_dir = os.path.join(OUTPUT_DIR, "_Plots_Global")
    os.makedirs(plot_dir, exist_ok=True)

    # Configuration for Plots
    FOLDER_ORDER = FOLDERS_S_FACTOR + FOLDERS_SENSITIVITY
    EXCLUDE_LIST = [] # Add substrings of sensors to exclude
    
    # Map folders to colors
    cmap = plt.get_cmap('tab10')
    folder_colors = {f: cmap(i) for i, f in enumerate(FOLDER_ORDER)}

    GROUP_STYLES = {
        'Platinum': {'marker': 'o', 'ls': '-',  'alpha': 0.9, 'lw': 2},
        'RhFe':     {'marker': 's', 'ls': '--', 'alpha': 0.8, 'lw': 2},
        'PtCo':     {'marker': '^', 'ls': ':',  'alpha': 0.8, 'lw': 2}
    }

    # --- Plot A: Global SNR Comparison ---
    plt.figure(figsize=(12, 8))
    global_has_snr = False
    
    for folder_key in FOLDER_ORDER:
        actual_folder = next((f for f in summary_storage.keys() if folder_key in f), None)
        if not actual_folder: continue
        
        for group_name, sensors in SENSOR_GROUPS.items():
            filtered_sensors = [s for s in sensors if not any(ex in s for ex in EXCLUDE_LIST)]
            avg_snr_list, currents_list = [], []
            
            for c in TARGET_CURRENTS:
                snr_key = f"SNR {c} mA"
                vals = []
                stats_list = summary_storage[actual_folder].get('Corr', [])
                for s_name in filtered_sensors:
                    s_data = next((item for item in stats_list if s_name in item['Sensor']), None)
                    if s_data and snr_key in s_data and not np.isnan(s_data[snr_key]):
                        vals.append(s_data[snr_key])
                
                if vals:
                    avg_snr_list.append(np.mean(vals))
                    currents_list.append(c)
            
            if avg_snr_list:
                global_has_snr = True
                style = GROUP_STYLES.get(group_name, {'marker': 'x', 'ls': '-'})
                plt.plot(currents_list, avg_snr_list, 
                         label=f"{group_name} @ {folder_key}",
                         color=folder_colors[folder_key],
                         marker=style['marker'], linestyle=style['ls'],
                         linewidth=style['lw'], alpha=style['alpha'], markersize=7)

    if global_has_snr:
        plt.title("Global SNR Comparison: Average per Group vs Temperature", fontsize=14)
        plt.xlabel("Excitation Current [mA]", fontsize=12); plt.ylabel("Average SNR [dB]", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "SNR_Global_Comparison.png"))
    plt.close()

    # --- Plot B: Global Noise Comparison (Log Scale) ---
    plt.figure(figsize=(12, 8))
    global_has_noise = False
    
    for folder_key in FOLDER_ORDER:
        actual_folder = next((f for f in summary_storage.keys() if folder_key in f), None)
        if not actual_folder: continue
        
        for group_name, sensors in SENSOR_GROUPS.items():
            filtered_sensors = [s for s in sensors if not any(ex in s for ex in EXCLUDE_LIST)]
            avg_noise_list, currents_list = [], []
            
            for c in TARGET_CURRENTS:
                noise_key = f"Szum {c} mA"
                vals = []
                stats_list = summary_storage[actual_folder].get('Corr', [])
                for s_name in filtered_sensors:
                    s_data = next((item for item in stats_list if s_name in item['Sensor']), None)
                    if s_data and noise_key in s_data and not np.isnan(s_data[noise_key]):
                        vals.append(s_data[noise_key])
                
                if vals:
                    avg_noise_list.append(np.mean(vals))
                    currents_list.append(c)
            
            if avg_noise_list:
                global_has_noise = True
                style = GROUP_STYLES.get(group_name, {'marker': 'x', 'ls': '-'})
                plt.plot(currents_list, avg_noise_list, 
                         label=f"{group_name} @ {folder_key}",
                         color=folder_colors[folder_key],
                         marker=style['marker'], linestyle=style['ls'],
                         linewidth=style['lw'], alpha=style['alpha'], markersize=7)

    if global_has_noise:
        plt.title("Global Noise Comparison: Average per Group vs Temperature", fontsize=14)
        plt.xlabel("Excitation Current [mA]", fontsize=12); plt.ylabel("Average Noise [Ohm]", fontsize=12)
        plt.yscale('log')
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Noise_Global_Comparison.png"))
    plt.close()

    print(f"--- FINISHED in {time.time()-start_t:.2f}s ---")

if __name__ == "__main__":
    main()
