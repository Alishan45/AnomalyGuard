"""
Dataset Generator for AnomalyGuard
Generates realistic financial transaction data with well-defined anomaly injection scenarios.
Note: Ground-truth anomaly labels are strictly for post-hoc evaluation and NEVER fed into
unsupervised models during training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_transaction_dataset(n_samples: int = 10000, anomaly_ratio: float = 0.025, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    # 1. Base timestamps over a 30-day window
    start_date = datetime(2026, 10, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(seconds=int(x)) for x in np.sort(np.random.uniform(0, 30 * 86400, n_samples))]
    
    # 2. Customers and Merchants
    n_customers = int(n_samples * 0.15)
    customer_ids = [f"CUST_{i:04d}" for i in np.random.randint(1000, 1000 + n_customers, n_samples)]
    
    categories = ['grocery', 'dining', 'utilities', 'transport', 'entertainment', 'electronics', 'travel', 'luxury_retail']
    category_weights = [0.35, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]
    merchant_categories = np.random.choice(categories, size=n_samples, p=category_weights)
    
    channels = ['in_store', 'pos_contactless', 'mobile_app', 'web', 'atm']
    channel_weights = [0.30, 0.35, 0.20, 0.10, 0.05]
    channels_sampled = np.random.choice(channels, size=n_samples, p=channel_weights)
    
    # 3. Normal feature distributions
    # Amount (log-normal, typically $15 to $120, median around $45)
    amounts = np.round(np.random.lognormal(mean=3.8, sigma=0.75, size=n_samples), 2)
    amounts = np.clip(amounts, 1.50, 850.00)
    
    # Distance from home (km, gamma distribution, typical 0.5 to 25 km)
    distance_home = np.round(np.random.gamma(shape=2.0, scale=4.0, size=n_samples), 2)
    
    # Time since last transaction (hours, exponential)
    time_since_last_txn = np.round(np.random.exponential(scale=18.0, size=n_samples), 2) + 0.1
    
    # Distance from last transaction location (km)
    distance_last_txn = np.round(np.random.gamma(shape=1.5, scale=3.0, size=n_samples), 2)
    
    # Device risk score (0.00 to 1.00; normal mostly low 0.01 - 0.25)
    device_risk = np.round(np.random.beta(a=1.5, b=12.0, size=n_samples), 3)
    
    # Failed PIN / authentication attempts in last 24h (mostly 0)
    failed_attempts = np.random.choice([0, 1, 2], size=n_samples, p=[0.94, 0.05, 0.01])
    
    # International transaction (binary)
    is_international = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    
    # Synthetic anomaly indicator (ground truth for benchmark evaluation only)
    is_anomaly = np.zeros(n_samples, dtype=int)
    anomaly_scenario = ["normal"] * n_samples
    
    # 4. Inject realistic anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(np.arange(n_samples), size=n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        scenario_type = np.random.choice([
            'high_amount_burst',
            'impossible_travel',
            'suspicious_overseas_luxury',
            'card_testing_velocity',
            'atm_credential_stuffing'
        ])
        is_anomaly[idx] = 1
        anomaly_scenario[idx] = scenario_type
        
        if scenario_type == 'high_amount_burst':
            # Extreme sudden purchase amount
            amounts[idx] = np.round(np.random.uniform(2500.0, 9500.0), 2)
            merchant_categories[idx] = np.random.choice(['luxury_retail', 'electronics'])
            channels_sampled[idx] = np.random.choice(['web', 'in_store'])
            device_risk[idx] = np.round(np.random.uniform(0.60, 0.95), 3)
            
        elif scenario_type == 'impossible_travel':
            # Long distance in unrealistically short time (>800 km/h)
            time_since_last_txn[idx] = np.round(np.random.uniform(0.05, 0.40), 2)  # 3 to 24 mins
            distance_last_txn[idx] = np.round(np.random.uniform(900.0, 4500.0), 2)  # thousands of km
            distance_home[idx] = np.round(np.random.uniform(1000.0, 5000.0), 2)
            is_international[idx] = 1
            channels_sampled[idx] = 'pos_contactless'
            
        elif scenario_type == 'suspicious_overseas_luxury':
            # International luxury purchase with high device risk
            amounts[idx] = np.round(np.random.uniform(1800.0, 6000.0), 2)
            is_international[idx] = 1
            merchant_categories[idx] = 'luxury_retail'
            device_risk[idx] = np.round(np.random.uniform(0.75, 0.99), 3)
            failed_attempts[idx] = np.random.choice([1, 2, 3])
            
        elif scenario_type == 'card_testing_velocity':
            # Rapid micro-transactions with high frequency
            amounts[idx] = np.round(np.random.uniform(0.50, 2.50), 2)
            time_since_last_txn[idx] = np.round(np.random.uniform(0.01, 0.05), 3)  # seconds to minutes
            channels_sampled[idx] = 'web'
            device_risk[idx] = np.round(np.random.uniform(0.55, 0.92), 3)
            
        elif scenario_type == 'atm_credential_stuffing':
            # Repeated failed attempts at ATM far from home
            channels_sampled[idx] = 'atm'
            amounts[idx] = np.round(np.random.uniform(400.0, 1200.0), 2)
            failed_attempts[idx] = np.random.choice([3, 4, 5])
            distance_home[idx] = np.round(np.random.uniform(80.0, 450.0), 2)
            device_risk[idx] = np.round(np.random.uniform(0.65, 0.98), 3)

    # 5. Build DataFrame
    df = pd.DataFrame({
        'transaction_id': [f"TXN_{i:06d}" for i in range(1, n_samples + 1)],
        'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'customer_id': customer_ids,
        'merchant_category': merchant_categories,
        'channel': channels_sampled,
        'amount': amounts,
        'distance_from_home_km': distance_home,
        'distance_from_last_txn_km': distance_last_txn,
        'time_since_last_txn_hours': time_since_last_txn,
        'device_risk_score': device_risk,
        'failed_attempts_last_24h': failed_attempts,
        'is_international': is_international,
        'is_synthetic_anomaly': is_anomaly,
        'anomaly_scenario': anomaly_scenario
    })
    
    return df

if __name__ == '__main__':
    df = generate_transaction_dataset(n_samples=10000, anomaly_ratio=0.025, random_seed=42)
    output_path = "data/transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions with {df['is_synthetic_anomaly'].sum()} anomalies ({df['is_synthetic_anomaly'].mean()*100:.2f}%) to {output_path}")
