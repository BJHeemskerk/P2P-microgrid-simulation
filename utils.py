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
    'works_home' : {'nacht': 5, 'ochtend': 20, 'middag': 25, 'avond': 30}
}

# Seizoensfactor per dag via interpolatie tussen maandwaarden
seizoensfactoren_maand = {
    1: 1.6, 2: 1.6, 3: 1.45, 4: 1.3, 5: 1.15, 6: 1,
    7: 1, 8: 1, 9: 1.15, 10: 1.3, 11: 1.45, 12: 1.6
}

profile_base_kwh = {
    'standaard': 6.8,
    'vroege_vogel': 6.8,
    'avondmens': 6.8,
    'zuinig': 4.8,  # lager verbruik
    'works_home': 9.1
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

def simulate_household_usage(n_days, profile, start_date=datetime(2024, 1, 1)):
    base_kwh = profile_base_kwh[profile]
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


def generate_household_dataframe(n_days=365, n_households=30, start_date=datetime(2024, 1, 1), profile_ratios=None):
    huishoudens = [f'household_{i+1}' for i in range(n_households)]
    profielen = list(profile_distributions.keys())
    
    # Als geen verhoudingen meegegeven zijn, kies willekeurig
    if profile_ratios is None:
        profile_ratios = {profiel: 1/len(profielen) for profiel in profielen}
    
    # Controle: som van verhoudingen moet 1 zijn
    if not np.isclose(sum(profile_ratios.values()), 1.0):
        raise ValueError("De som van alle profielverhoudingen moet gelijk zijn aan 1.0")

    # Bepaal hoeveel huishoudens per profiel
    n_per_profiel = {profiel: int(round(ratio * n_households)) for profiel, ratio in profile_ratios.items()}

    # Corrigeer afronding zodat totaal klopt
    totaal_toegewezen = sum(n_per_profiel.values())
    verschil = n_households - totaal_toegewezen
    if verschil != 0:
        # Corrigeer bij het eerste profiel (klein verschil)
        eerste_profiel = list(n_per_profiel.keys())[0]
        n_per_profiel[eerste_profiel] += verschil

    # Genereer lijst van toegewezen profielen
    toegewezen_profielen = []
    for profiel, aantal in n_per_profiel.items():
        toegewezen_profielen.extend([profiel] * aantal)
    
    # Shuffle zodat het niet gesorteerd is op profiel
    random.shuffle(toegewezen_profielen)

    records = []

    for huishouden, profiel in zip(huishoudens, toegewezen_profielen):
        usage = simulate_household_usage(n_days, profiel, start_date=start_date)
        for day_idx, dag_data in enumerate(usage):
            datum = start_date + timedelta(days=day_idx)
            record = {
                'household': huishouden,
                'profiel': profiel,
                'datum': datum.strftime('%d-%m-%Y'),
                'totaal_kWh': sum(dag_data)
            }
            for uur in range(24):
                record[f"{uur}"] = dag_data[uur]
            records.append(record)

    df = pd.DataFrame(records).set_index("datum")
    return df

