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

# Seasonal factoring for each month
seasonfactoring_month = {
    1: 1.6,
    2: 1.6,
    3: 1.45,
    4: 1.3,
    5: 1.15,
    6: 1,
    7: 1,
    8: 1,
    9: 1.15,
    10: 1.3,
    11: 1.45,
    12: 1.6
}

# Average daily consumption (kWh) base values for each profile
profile_base_kwh = {
    'base_profile': 6.8,
    'early_bird': 6.8,
    'evening_person': 6.8,
    'night_owl': 7.8,
    'energy_saver': 4.8,
    'works_home': 9.1
}


class Battery:
    """
    Simulates a battery storage system with charging/discharging capabilities.
    """

    def __init__(self, capacity_kwh, c_rate, efficiency):
        """
        Initialize battery.

        Parameters:
        -----------
        capacity_kwh : float
            Battery capacity in kWh.
        c_rate : float
            Maximum charge/discharge rate as a fraction of capacity per hour.
        efficiency : float
            Round-trip efficiency (0-1).
        """
        self.capacity = capacity_kwh
        self.max_discharge = self.capacity * c_rate
        self.efficiency = efficiency
        self.state_of_charge = 0.0  # Initial state of charge in kWh

    def charge(self, power_kw, duration_hr):
        """
        Charge the battery with given power and duration.

        Parameters:
        -----------
        power_kw : float
            Charging power in kW.
        duration_hr : float
            Duration of charging in hours.
        """
        # Limit charging power to max discharge rate (assumed symmetric)
        power = min(power_kw, self.max_discharge)

        # Calculate effective energy added considering efficiency losses
        energy_added = power * duration_hr * self.efficiency

        # Update state of charge without exceeding capacity
        self.state_of_charge = min(
            self.state_of_charge + energy_added, self.capacity
            )


    def discharge(self, power_kw, duration_hr):
        """
        Discharge the battery with given power and duration.

        Parameters:
        -----------
        power_kw : float
            Discharging power in kW.
        duration_hr : float
            Duration of discharging in hours.
        """
        # Limit discharging power to max discharge rate
        power = min(power_kw, self.max_discharge)

        # Calculate effective energy removed considering efficiency losses
        energy_removed = power * duration_hr / self.efficiency

        # Update state of charge without going below zero
        self.state_of_charge = max(
            self.state_of_charge - energy_removed, 0.0
            )

    @property
    def soc_percent(self):
        """Return state of charge as a percentage."""
        return 100 * (self.state_of_charge / self.capacity)


def get_seasonal_factor_for_day(current_date):
    """
    Calculate the seasonal consumption factor for a specific day using
    linear interpolation between monthly factors.

    Parameters:
    -----------
    current_date : datetime
        The date for which to calculate the seasonal factor.

    Returns:
    --------
    float
        Seasonal factor multiplier.
    """
    # Get current date and next month
    day_of_month = current_date.day
    month = current_date.month
    next_month = 1 if month == 12 else month + 1

    # Get seasonal factors for current and next month
    start_factor = seasonfactoring_month[month]
    end_factor = seasonfactoring_month[next_month]

    # Calculate the number of days in current month
    days_in_month = (
        datetime(
            current_date.year
            if next_month != 1
            else current_date.year + 1,
            next_month, 1
            ) - datetime(current_date.year, month, 1)
        ).days
    
    # Linear interpolation factor for the day within the month
    interpolation_factor = day_of_month / days_in_month

    # Interpolated seasonal factor for the day
    return start_factor + (end_factor - start_factor) * interpolation_factor


def simulate_household_usage(n_days, profile, start_date=datetime(2024, 1, 1)):
    """
    Simulate hourly electricity consumption for a household over multiple days
    based on profile usage patterns, seasonal factors, noise, and random outliers.

    Parameters:
    -----------
    n_days : int
        Number of days to simulate.
    profile : str
        Household consumption profile key.
    start_date : datetime, optional
        Start date of simulation (default: Jan 1, 2024).

    Returns:
    --------
    list of list of float
        Hourly electricity consumption values per day.
    """
    base_kwh = profile_base_kwh[profile]
    profile_weights = profile_distributions[profile]
    usage_data = []

    # Generate random outlier days (1-2 days per week)
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

    # Simulate usage for each day and hour
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        seasonal = get_seasonal_factor_for_day(current_date)
        hourly_usage = []

        for hour in range(24):
            # Determine time block for hour
            if 0 <= hour < 6:
                block = 'night'
            elif 6 <= hour < 12:
                block = 'morning'
            elif 12 <= hour < 18:
                block = 'afternoon'
            else:
                block = 'evening'

            # Calculate base usage per hour in this block
            block_percentage = profile_weights[block]
            block_hours = 6  # hours per block
            base_usage = base_kwh * (block_percentage / 100) / block_hours

            # Add noise (normally distributed multiplier ~ N(1.0, 0.1))
            noise = np.random.normal(1.0, 0.1)

            # Final hourly usage factoring seasonal variation and noise
            usage = base_usage * noise * seasonal
            hourly_usage.append(usage)

        # Add random outlier spike on some days (between 6 and 22 hrs)
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
    """
    Generate a DataFrame containing simulated hourly consumption data
    for multiple households over a number of days, based on specified profile ratios.

    Parameters:
    -----------
    n_days : int, optional
        Number of days to simulate (default 365).
    n_households : int, optional
        Number of households (default 30).
    start_date : datetime, optional
        Simulation start date (default Jan 1, 2024).
    profile_ratios : dict, optional
        Dictionary with profile names as keys and ratio (sum to 1.0) as values.
        If None, profiles are evenly distributed.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by date with hourly consumption per household and profile.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Name households and find profiles
    households = [f'household_{i+1}' for i in range(n_households)]
    profiles = list(profile_distributions.keys())

    # Default to equal distribution if none specified
    if profile_ratios is None:
        profile_ratios = {profiel: 1/len(profiles) for profiel in profiles}

    # Check sum of ratios equals 1
    if not np.isclose(sum(profile_ratios.values()), 1.0):
        raise ValueError(
            "The sum of the profile distribution needs to be 1.0."
            )

    # Calculate households per profile (rounded)
    n_per_profiel = {
        profile: int(round(ratio * n_households))
        for profile, ratio
        in profile_ratios.items()
        }

    # Correct rounding to ensure total matches n_households
    total_assigned = sum(n_per_profiel.values())
    difference = n_households - total_assigned
    if difference != 0:
        # Correct small errors
        first_profile = list(n_per_profiel.keys())[0]
        n_per_profiel[first_profile] += difference

    # Build profile assignment list for all households
    assigned_profiles = []
    for profile, amount in n_per_profiel.items():
        assigned_profiles.extend([profile] * amount)

    # Shuffle to avoid sorted profiles by household ID
    random.shuffle(assigned_profiles)

    records = []

    # Simulate usage and build records for each household and day
    for household, profile in zip(households, assigned_profiles):
        usage = simulate_household_usage(
            n_days, profile, start_date=start_date
            )
        for day_idx, dag_data in enumerate(usage):
            datum = start_date + timedelta(days=day_idx)
            record = {
                'household': household,
                'profiel': profile,
                'datum': datum.strftime('%d-%m-%Y'),
                'totaal_kWh': sum(dag_data)
            }
            # Add hourly usage fields (0-23)
            for uur in range(24):
                record[f"{uur}"] = dag_data[uur]
            records.append(record)

    # Create DataFrame and index by date string
    df = pd.DataFrame(records).set_index("datum")
    return df


def generate_grid_prize_data(
        n_days=365,
        start_date=datetime(2024, 1, 1),
        base_price=0.1998,
        solar_csv_path="data/solar_strength.csv",
        seed=None
):
    """
    Generate hourly grid electricity prices for each day over a period,
    incorporating trends, seasonality, solar production influence, and hourly variations.

    Parameters:
    -----------
    n_days : int, optional
        Number of days to generate prices for (default 365).
    start_date : datetime, optional
        Start date of price generation (default Jan 1, 2024).
    base_price : float, optional
        Base grid electricity price in EUR/kWh (default 0.1998).
    solar_csv_path : str, optional
        Path to CSV file containing solar production strength data.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by date string with columns for each hour (0-23) containing prices.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    solar_df = pd.read_csv(solar_csv_path, index_col="DATE", parse_dates=True)

    # Normalize solar values hourly per day
    solar_df = solar_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

    data = []

    for day_offset in range(n_days):
        current_datetime = start_date + timedelta(days=day_offset)
        date_str = current_datetime.strftime("%d-%m-%Y")

        # Calculate month index for trend/seasonality calculations
        days_since_start = (current_datetime - start_date).days
        continuous_month_index = days_since_start / 30

        # Price trend decreases slowly over time (logarithmic decay)
        trend = -0.01 * math.log1p(continuous_month_index + 1)

        # Seasonal variation using cosine function (annual cycle)
        seasonal = 0.005 * math.cos((2 * math.pi / 12) * continuous_month_index)

        # Retrieve solar strength for the day, default 0 if missing
        if current_datetime in solar_df.index:
            solar_row = solar_df.loc[current_datetime]
        else:
            solar_row = pd.Series([0] * 24, index=range(24))

        hourly_prices = {}

        for hour in range(24):
            # Daily pattern: lower at night, higher at morning/evening peak
            if 7 <= hour <= 10 or 17 <= hour <= 21:
                hourly_variation = 0.01
            elif 0 <= hour < 7 or hour > 23:
                hourly_variation = -0.005
            else:
                hourly_variation = 0

            # Solar influence: cheaper electricity during high solar generation (daytime)
            solar_strength = solar_row.get(hour, 0)
            solar_influence = -0.01 * solar_strength

            # Calculating and rounding the price
            price = round(
                base_price + trend + seasonal + solar_influence + hourly_variation,
                4
            )
            hourly_prices[str(hour)] = price

        # Adding prices to dict, then appending to list
        hourly_prices["date"] = date_str
        data.append(hourly_prices)

    # Returning the dataframe
    return pd.DataFrame(data).set_index("date")
