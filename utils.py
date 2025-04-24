import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Profielen en verdelingen
profile_distributions = {
    'standaard': {'nacht': 5, 'ochtend': 25, 'middag': 30, 'avond': 40},
    'vroege_vogel': {'nacht': 5, 'ochtend': 35, 'middag': 30, 'avond': 30},
    'avondmens': {'nacht': 5, 'ochtend': 15, 'middag': 30, 'avond': 50},
    'zuinig': {'nacht': 5, 'ochtend': 20, 'middag': 25, 'avond': 30},
}

def simulate_household_usage(n_days, profile, base_kwh=6.8):
    profile_weights = profile_distributions[profile]
    usage_data = []

    # Genereer outliers per week
    weeks = range(0, n_days, 7)
    outlier_days = []
    for w_start in weeks:
        week_days = list(range(w_start, min(w_start + 7, n_days)))
        max_outliers = min(2, len(week_days))
        outlier_count = random.randint(1, max_outliers) if max_outliers > 0 else 0
        if outlier_count > 0:
            outlier_days.extend(random.sample(week_days, k=outlier_count))

    for day in range(n_days):
        hourly_usage = []
        for hour in range(24):
            if 0 <= hour < 6:
                block = 'nacht'
            elif 6 <= hour < 12:
                block = 'ochtend'
            elif 12 <= hour < 18:
                block = 'middag'
            else:
                block = 'avond'

            block_percentage = profile_weights[block]
            block_hours = 6
            base_usage = base_kwh * (block_percentage / 100) / block_hours
            noise = np.random.normal(1.0, 0.1)
            usage = base_usage * noise
            hourly_usage.append(usage)

        # Voeg outlier toe
        if day in outlier_days:
            outlier_hour = random.randint(6, 22)
            hourly_usage[outlier_hour] += random.uniform(1.5, 3.0)

        usage_data.append(hourly_usage)

    return usage_data

def generate_household_dataframe(n_days=365, n_households=30, start_date=datetime(2024, 1, 1)):
    huishoudens = [f'household_{i+1}' for i in range(n_households)]
    profielen = list(profile_distributions.keys())
    records = []

    for huishouden in huishoudens:
        profiel = random.choice(profielen)
        usage = simulate_household_usage(n_days, profiel)
        for day_idx, dag_data in enumerate(usage):
            datum = start_date + timedelta(days=day_idx)
            record = {
                'household': huishouden,
                'profiel': profiel,
                'datum': datum.strftime("%d-%m-%Y"),
                'totaal_kWh': sum(dag_data)
            }
            for uur in range(24):
                record[f"{uur}"] = dag_data[uur]
            records.append(record)

    df = pd.DataFrame(records).set_index("datum")
    return df
