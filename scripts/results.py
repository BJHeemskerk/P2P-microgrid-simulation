import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

from scripts.utils import (
    generate_household_dataframe, generate_grid_prize_data
    )
from scripts.model import MicroGrid


PROFILE_COLORS = {
        'early_bird': '#ff0000',
        'evening_person': '#00ff00',
        'energy_saver': '#0000ff',
        'base_profile': '#ffff00',
        'night_owl': '#ff00ff',
        'works_home': '#00ffff'
    }


def grid_price_vs_local_price(sim_df, save=False, folder="simulation"):
    sim_df["day"] = pd.to_datetime(sim_df["day"], format="%d-%m-%Y")

    sim_df["timestamp"] = sim_df["day"] + pd.to_timedelta(
        sim_df["hour"], unit="h"
        )

    plt.figure(figsize=(14, 5))
    for col in ["grid_price", "local_price"]:
        plt.plot(sim_df["timestamp"], sim_df[col], label=col)

    plt.xlabel("Time")
    plt.ylabel("Energy Price")
    plt.title("Grid vs Local Energy Price Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(
            f"results/{folder}/Price_comparison.png",
            dpi=300
            )
        plt.close()
    else:
        plt.show()


def impact_of_battery_usage(sim_df, save=False, folder="simulation"):
    sim_df["battery_used"] = sim_df["battery_state"].apply(
        lambda soc: "Battery Used" if soc > 0 else "No Battery"
        )

    plt.figure(figsize=(14, 6))
    sns.boxplot(y="battery_used", x="local_price", data=sim_df)
    plt.title("Local Energy Price Distribution With vs. Without Battery Usage")
    plt.ylabel("Battery Usage")
    plt.xlabel("Local Energy Price (€/kWh)")
    plt.grid(True)

    if save:
        plt.savefig(
            f"results/{folder}/Impact_battery.png",
            dpi=300
            )
        plt.close()
    else:
        plt.show()


def energy_delta_per_agent(agent_df, save=False, folder="simulation"):
    agent_summary = agent_df.groupby("agent_id").agg({
        "profile": "first",
        "produced": "sum",
        "consumed": "sum"
    }).reset_index()
    agent_summary["net_balance"] = agent_summary["produced"] \
        - agent_summary["consumed"]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=agent_summary.sort_values("net_balance"),
        x="agent_id", y="net_balance", palette=PROFILE_COLORS,
        hue="profile"
        )
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Net Energy Balance Per Agent (Produced - Consumed)")
    plt.xlabel("Agent ID")
    plt.ylabel("Net Energy Balance (kWh)")
    plt.tight_layout()

    if save:
        plt.savefig(
            f"results/{folder}/Agent_net_energy.png",
            dpi=300
            )
        plt.close()
    else:
        plt.show()


def plot_import_export_per_agent(agent_df, save=False, folder="simulation"):
    # Gemiddeld profiel per agent (moet 1 profiel per agent zijn)
    agent_profiles = agent_df.groupby(
        'agent_id'
        )['profile'].first().reset_index()

    # Sommeer import/export per agent
    grouped = agent_df.groupby(
        'agent_id'
        )[['import', 'export']].sum().reset_index()

    # Voeg profiel toe aan gegroepeerde data
    grouped = grouped.merge(agent_profiles, on='agent_id')

    # Plot instellen
    x = np.arange(len(grouped))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))

    # Achtergrondkleuren per profiel
    for i, row in grouped.iterrows():
        color = PROFILE_COLORS.get(row['profile'], '#eeeeee')
        ax.axvspan(i - 0.4, i + 0.4, color=color, alpha=0.3)

    # Staafdiagrammen
    ax.bar(
        x - width/2, grouped['import'],
        width, label='Import', color='red'
        )
    ax.bar(
        x + width/2, grouped['export'],
        width, label='Export', color='green'
        )

    # X-as instellingen
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['agent_id'])
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Totaal Traded Energy (kWh)')
    ax.set_title('Import en Export per Agent met Profielachtergrond')
    ax.legend()

    # Legenda voor profielen (kleurvakken)
    legend_patches = [
        Patch(facecolor=color, edgecolor='none', label=profile)
        for profile, color
        in PROFILE_COLORS.items()
        ]
    ax.legend(
        handles=[
            *legend_patches,
            Patch(color='red', label='Import'),
            Patch(color='green', label='Export')
            ]
        )

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save:
        plt.savefig(
            f"results/{folder}/Import_export.png",
            dpi=300
            )
        plt.close()
    else:
        plt.show()


def show_sim_report(sim, profile_ratios, save=False, folder="simulation"):
    unique_profiles = sim.agent_df['profile'].unique().tolist()

    agent_summary = sim.agent_df.groupby("profile")[
        [
            "produced", "consumed", "import",
            "export", "energy_centralgrid",
            "energy_microgrid"
        ]
    ].mean().reset_index()
    agent_summary["net_balance"] = agent_summary["produced"] - \
        agent_summary["consumed"]

    profile_ratio_lines = []
    for profile in unique_profiles:
        ratio = profile_ratios.get(profile, 'N/A')
        profile_ratio_lines.append(f"- {profile}: ratio = {ratio}")

    profile_ratio = "\n          ".join(profile_ratio_lines)

    profile_report_lines = []
    for profile in unique_profiles:
        mask_df = agent_summary[agent_summary["profile"] == profile]
        produced = mask_df["produced"].values[0]
        consumed = mask_df["consumed"].values[0]
        import_ = mask_df["import"].values[0]
        export_ = mask_df["export"].values[0]
        from_grid = mask_df["energy_centralgrid"].values[0]
        from_microgrid = mask_df["energy_microgrid"].values[0]
        net_balance = mask_df["net_balance"].values[0]
        profile_report_lines.append(f"- {profile}:")
        profile_report_lines.append(
            f"        - Avg Produced:     {produced:.2f} kW"
            )
        profile_report_lines.append(
            f"        - Avg Consumed:     {consumed:.2f} kW"
            )
        profile_report_lines.append(
            f"        - Avg Imported:     {import_:.2f} kW"
            )
        profile_report_lines.append(
            f"        - Avg Exported:     {export_:.2f} kW"
            )
        profile_report_lines.append(
            f"        - Avg From Grid:    {from_grid:.2f} kW"
            )
        profile_report_lines.append(
            f"        - Avg From Micro:   {from_microgrid:.2f} kW"
            )
        profile_report_lines.append(
            f"        - Net Balance:      {net_balance:.2f} kW"
            )

    profile_report = "\n            ".join(profile_report_lines)

    total_demand = sim.sim_df["microgrid_demand"].sum()
    total_supply = sim.sim_df["microgrid_supply"].sum()
    avg_energy_delta = sim.sim_df["energy_delta"].mean()
    avg_central_price = sim.sim_df["grid_price"].mean()
    min_central_price = sim.sim_df["grid_price"].min()
    max_central_price = sim.sim_df["grid_price"].max()
    avg_micro_price = sim.sim_df["local_price"].mean()
    min_micro_price = sim.sim_df["local_price"].min()
    max_micro_price = sim.sim_df["local_price"].max()
    std_micro_price = sim.sim_df["local_price"].std()

    report_text = f"""
        Details of the simulation:

          Parameters:
          ---------------------------------------------------

          - Number of Households:   {sim.num_agents}
          - Gini:                   {sim.gini:.2f}
          - Mean panels:            {sim.mean_panels:.2f} panels
          - Panel efficiency:       {sim.panel_efficiency:.2f} (%)
          - Battery capacity:       {sim.bat_capacity:.2f} kWh
          - Battery charge rate:    {sim.bat_c_rate:.2f} kW
          - Battery efficiency:     {sim.bat_efficiency:.2f} (%)
          - Number of days:         {len(sim.simulation_data)}

          Profiles used:
          ---------------------------------------------------
          {profile_ratio}

          Results:
          ---------------------------------------------------

            Energy statistics:
            -------------------------------------------------
            - Total demand:             {total_demand:.2f} kWh
            - Total supply:             {total_supply:.2f} kWh
            - Average energy delta:     {avg_energy_delta:.4f} kWh
            - Average grid price:       {avg_central_price:.4f} €/kWh
            - Min grid price:           {min_central_price:.4f} €/kWh
            - Max grid price:           {max_central_price:.4f} €/kWh
            - Average local price:      {avg_micro_price:.4f} €/kWh
            - Min local price:          {min_micro_price:.4f} €/kWh
            - Max local price:          {max_micro_price:.4f} €/kWh
            - Local price deviation:    {std_micro_price:.4f} €/kWh

            Profiles:
            -------------------------------------------------
            {profile_report}

    """
    if save:
        with open(f"results/{folder}/simulation_report.txt", "w") as file:
            file.write(report_text)
    else:
        print(report_text)


def gather_results(params, folder):
    results_path = "results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    folder_path = os.path.join(results_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Creating the dataframes for the simulation
    household_prosumption_data = generate_household_dataframe(
        n_days=params["n_days"],
        n_households=params["n_households"],
        profile_ratios=params["profile_ratios"],
        seed=42
        )

    production_data = pd.read_csv("data/solar_strength.csv", index_col="DATE")

    grid_price_data = generate_grid_prize_data(
        n_days=params["n_days"],
        seed=42
        )

    simulation = MicroGrid(
        n_households=params["n_households"],
        consumption_data=household_prosumption_data,
        production_data=production_data,
        grid_prize_data=grid_price_data,
        gini=params["gini"],
        mean_panels=params["mean_panels"],
        panel_efficiency=params["panel_efficiency"],
        bat_capacity=params["battery_capacity"],
        bat_c_rate=params["battery_charge_rate"],
        bat_efficiency=params["battery_efficiency"],
        seed=42,
        verbose=0
    )

    simulation.long_step(
        n=(params["n_days"] - 1) * 24
        )

    simulation._convert_to_dataframe()

    simulation.sim_df.to_csv(
        f"results/{folder}/simulation_data.csv", sep=";"
    )

    simulation.agent_df.to_csv(
        f"results/{folder}/agent_data.csv", sep=";"
    )

    grid_price_vs_local_price(simulation.sim_df, save=True, folder=folder)
    impact_of_battery_usage(simulation.sim_df, save=True, folder=folder)
    energy_delta_per_agent(simulation.agent_df, save=True, folder=folder)
    plot_import_export_per_agent(simulation.agent_df, save=True, folder=folder)
    show_sim_report(
        simulation, params["profile_ratios"],
        save=True, folder=folder
        )

    print(f"""\n
    Results have been gathered!

    You can find the gathered results in the folder results/{folder}
""")
