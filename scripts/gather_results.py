from results import gather_results


# Profile ratios, to turn off a profile you can set it to 0
profile_ratios = {
    'base_profile': 1/6,
    'early_bird': 1/6,
    'evening_person': 1/6,
    'night_owl': 1/6,
    'energy_saver': 1/6,
    'works_home': 1/6
}

# Parameters for the simulation
param_dict = {
    "Test-run 1": {
        "n_days": 366,
        "n_households": 30,
        "profile_ratios": profile_ratios,
        "gini": 0.1,
        "mean_panels": 16,
        "panel_efficiency": 0.2,
        "battery_capacity": 1000,
        "battery_charge_rate": 1,
        "battery_efficiency": 0.9
    }
}

if __name__ == "__main__":
    for key, value in param_dict.items():
        gather_results(
            params=value,
            folder=key
        )