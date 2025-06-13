import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math

# All profiles and distributions (added up they are 100)
profile_distributions = {
    'base_profile': {
        'night': 5, 'morning': 25, 'afternoon': 30, 'evening': 40
        },
    'early_bird': {
        'night': 5, 'morning': 35, 'afternoon': 30, 'evening': 30
        },
    'evening_person': {
        'night': 5, 'morning': 15, 'afternoon': 30, 'evening': 50
        },
    'night_owl': {
        'night': 45, 'morning': 5, 'afternoon': 15, 'evening': 35
        },
    'energy_saver': {
        'night': 5, 'morning': 25, 'afternoon': 30, 'evening': 40
        },
    'works_home': {
        'night': 5, 'morning': 25, 'afternoon': 35, 'evening': 35
        }
}

# Seasonal factoring for each month (for consumption increase)
seizoensfactoren_maand = {
    1: 1.6, 2: 1.6, 3: 1.45, 4: 1.3, 5: 1.15, 6: 1,
    7: 1, 8: 1, 9: 1.15, 10: 1.3, 11: 1.45, 12: 1.6
}

# Base kWh for average daily power usage per profile
profile_base_kwh = {
    'base_profile': 6.8,
    'early_bird': 6.8,
    'evening_person': 6.8,
    'night_owl': 7.8,
    'energy_saver': 4.8,
    'works_home': 9.1
}


class Battery:
    def __init__(self, capacity_kwh, c_rate, efficiency):
        self.capacity = capacity_kwh
        self.max_discharge = self.capacity * c_rate
        self.efficiency = efficiency
        self.state_of_charge = 0.0
        self.cost_of_battery = self.capacity * 750

    def charge(self, power_kw, duration_hr):
        # Limit the charging power to the maximum allowed
        power = min(power_kw, self.max_discharge)

        # Calculate the energy added, considering efficiency
        energy_added = power * duration_hr * self.efficiency

        # Update the soc without exceeding capacity
        self.state_of_charge = min(
            (self.state_of_charge + energy_added), self.capacity
            )

    def discharge(self, power_kw, duration_hr):
        # Limit the discharging power to the maximum allowed
        power = min(power_kw, self.max_discharge)

        # Calculate the energy removed, considering efficiency
        energy_removed = power * duration_hr / self.efficiency

        # Update the soc without going below zero
        self.state_of_charge = max(self.state_of_charge - energy_removed, 0.0)

    @property
    def soc_percent(self):
        return 100 * (self.state_of_charge / self.capacity)


def get_seasonal_factor_for_day(current_date):
    dag_van_maand = current_date.day
    maand = current_date.month
    volgende_maand = 1 if maand == 12 else maand + 1

    start_factor = seizoensfactoren_maand[maand]
    end_factor = seizoensfactoren_maand[volgende_maand]

    dagen_in_maand = (
        datetime(
            current_date.year
            if volgende_maand != 1
            else current_date.year + 1,
            volgende_maand, 1
            ) - datetime(current_date.year, maand, 1)).days
    interpolatie_factor = dag_van_maand / dagen_in_maand

    return start_factor + (end_factor - start_factor) * interpolatie_factor


def simulate_household_usage(n_days, profile, start_date=datetime(2024, 1, 1)):
    base_kwh = profile_base_kwh[profile]
    profile_weights = profile_distributions[profile]
    usage_data = []

    # Generate outliers for every week
    weeks = range(0, n_days, 7)
    outlier_days = []
    for w_start in weeks:
        week_days = list(range(w_start, min(w_start + 7, n_days)))
        max_outliers = min(2, len(week_days))
        outlier_count = random.randint(
            1, max_outliers
            ) if max_outliers > 0 else 0
        if outlier_count > 0:
            outlier_days.extend(random.sample(week_days, k=outlier_count))

    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        seasonal = get_seasonal_factor_for_day(current_date)
        hourly_usage = []

        for hour in range(24):
            if 0 <= hour < 6:
                block = 'night'
            elif 6 <= hour < 12:
                block = 'morning'
            elif 12 <= hour < 18:
                block = 'afternoon'
            else:
                block = 'evening'

            block_percentage = profile_weights[block]
            block_hours = 6
            base_usage = base_kwh * (block_percentage / 100) / block_hours
            noise = np.random.normal(1.0, 0.1)
            usage = base_usage * noise * seasonal
            hourly_usage.append(usage)

        # Add outliers
        if day in outlier_days:
            outlier_hour = random.randint(6, 22)
            hourly_usage[outlier_hour] += random.uniform(1.5, 3.0)

        usage_data.append(hourly_usage)

    return usage_data


def generate_household_dataframe(
        n_days=365, n_households=30,
        start_date=datetime(2024, 1, 1),
        profile_ratios=None,
        seed=None
        ):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    huishoudens = [f'household_{i+1}' for i in range(n_households)]
    profielen = list(profile_distributions.keys())

    # Als geen verhoudingen meegegeven zijn, kies willekeurig
    if profile_ratios is None:
        profile_ratios = {profiel: 1/len(profielen) for profiel in profielen}

    # Controle: som van verhoudingen moet 1 zijn
    if not np.isclose(sum(profile_ratios.values()), 1.0):
        raise ValueError(
            "De som van alle profielverhoudingen moet gelijk zijn aan 1.0"
            )

    # Bepaal hoeveel huishoudens per profiel
    n_per_profiel = {
        profiel:
        int(round(ratio * n_households))
        for profiel, ratio
        in profile_ratios.items()
        }

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
        usage = simulate_household_usage(
            n_days, profiel, start_date=start_date
            )
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


def generate_grid_prize_data(
        n_days=365,
        start_date=datetime(2024, 1, 1),
        base_price=0.1998,
        solar_csv_path="data/solar_strength.csv",
        seed=None
        ):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    solar_df = pd.read_csv(solar_csv_path, index_col="DATE", parse_dates=True)

    # Normalize solar values hourly per day
    solar_df = solar_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

    data = []
    drift = 0

    for day_offset in range(n_days):
        current_datetime = start_date + timedelta(days=day_offset)
        date_str = current_datetime.strftime("%d-%m-%Y")

        days_since_start = (current_datetime - start_date).days
        continuous_month_index = days_since_start / 30

        trend = -0.01 * math.log1p(continuous_month_index + 1)
        seasonal = 0.005 * math.cos((2 * math.pi / 12) * continuous_month_index)

        if current_datetime in solar_df.index:
            solar_row = solar_df.loc[current_datetime]
        else:
            solar_row = pd.Series([0] * 24, index=range(24))

        hourly_prices = {}

        for hour in range(24):
            # Daily pattern: lower at night, higher at morning/evening peak
            if 6 <= hour <= 9 or 17 <= hour <= 20:
                hourly_variation = 0.01
            elif 0 <= hour < 6 or hour > 21:
                hourly_variation = -0.005
            else:
                hourly_variation = 0

            # Solar influence: cheaper electricity during high solar generation (daytime)
            solar_strength = solar_row.get(hour, 0)
            solar_influence = -0.01 * solar_strength

            price = round(
                base_price + trend + seasonal + drift + solar_influence + hourly_variation,
                4
            )
            hourly_prices[str(hour)] = price

        hourly_prices["date"] = date_str
        data.append(hourly_prices)

    return pd.DataFrame(data).set_index("date")
