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

# Seizoensfactor per dag via interpolatie tussen maandwaarden
seizoensfactoren_maand = {
    1: 1.2, 2: 1.2, 3: 1.0, 4: 0.95, 5: 0.95, 6: 0.85,
    7: 0.85, 8: 0.90, 9: 1.0, 10: 1.05, 11: 1.05, 12: 1.2
}

def get_seasonal_factor_for_day(current_date):
    dag_van_maand = current_date.day
    maand = current_date.month
    volgende_maand = 1 if maand == 12 else maand + 1

    start_factor = seizoensfactoren_maand[maand]
    end_factor = seizoensfactoren_maand[volgende_maand]

    dagen_in_maand = (datetime(current_date.year if volgende_maand != 1 else current_date.year + 1, volgende_maand, 1) - datetime(current_date.year, maand, 1)).days
    interpolatie_factor = dag_van_maand / dagen_in_maand

    return start_factor + (end_factor - start_factor) * interpolatie_factor

def simulate_household_usage(n_days, profile, base_kwh=6.8, start_date=datetime(2024, 1, 1)):
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
        current_date = start_date + timedelta(days=day)
        seasonal = get_seasonal_factor_for_day(current_date)
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
            usage = base_usage * noise * seasonal
            hourly_usage.append(usage)

        # Voeg outlier toe
        if day in outlier_days:
            outlier_hour = random.randint(6, 22)
            hourly_usage[outlier_hour] += random.uniform(1.5, 3.0)

        usage_data.append(hourly_usage)

    return usage_data

def generate_household_dataframe(n_days=365, n_households=30, start_date=datetime(2024, 1, 1)):
    huishoudens = [f'huishouden_{i+1}' for i in range(n_households)]
    profielen = list(profile_distributions.keys())
    records = []

    for huishouden in huishoudens:
        profiel = random.choice(profielen)
        usage = simulate_household_usage(n_days, profiel, start_date=start_date)
        for day_idx, dag_data in enumerate(usage):
            datum = start_date + timedelta(days=day_idx)
            record = {
                'huishouden': huishouden,
                'profiel': profiel,
                'datum': datum.strftime('%Y-%m-%d'),
                'totaal_kWh': sum(dag_data)
            }
            for uur in range(24):
                record[f'{uur:02d}:00'] = dag_data[uur]
            records.append(record)

    df = pd.DataFrame(records)
    return df

