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


# Establishing profile colors for graphs
PROFILE_COLORS = {
        'early_bird': '#ff0000',
        'evening_person': '#00ff00',
        'energy_saver': '#0000ff',
        'base_profile': '#ffff00',
        'night_owl': '#ff00ff',
        'works_home': '#00ffff'
    }


def grid_price_vs_local_price(sim_df, save=False, folder="simulation"):
    """
    Plot daily average grid price, local energy price, and calculated price over time.

    Parameters:
    -----------
    sim_df : pd.DataFrame
        DataFrame containing simulation data with columns ['day', 'hour', 'grid_price', 'local_price', 'calculated_price'].
    save : bool, optional
        Whether to save the plot as an image file. Default is False.
    folder : str, optional
        Folder name inside 'results' where to save the plot. Default is 'simulation'.
    """
    # Convert 'day' to datetime and create timestamp column for plotting
    sim_df["day"] = pd.to_datetime(sim_df["day"], format="%d-%m-%Y")
    sim_df["timestamp"] = sim_df["day"] + pd.to_timedelta(sim_df["hour"], unit="h")

    # Compute daily average of prices
    daily_avg = sim_df.groupby("day")[["grid_price", "local_price", "calculated_price"]].mean().reset_index()

    # Define colors for each price type
    colors = {
        "grid_price": "#1f77b4",
        "local_price": "#2ca02c",
        "calculated_price": "#d62728"
    }

    plt.figure(figsize=(14, 5))

    # Plot all three price types
    for col in ["calculated_price", "local_price", "grid_price"]:
        plt.plot(daily_avg["day"], daily_avg[col], label=col.replace('_', ' ').title(), color=colors[col], alpha=0.8)

    # Setting axis and title
    plt.xlabel("Date")
    plt.ylabel("Average Energy Price")
    plt.title("Daily Average Grid, Local, and Calculated Energy Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save or display the plot
    if save:
        plt.savefig(f"results/{folder}/Daily_Price_comparison.png", dpi=300)
        plt.close()
    else:
        plt.show()


def impact_of_battery_usage(sim_df, save=False, folder="simulation"):
    """
    Visualize the effect of battery usage on the distribution of local energy prices.

    Parameters:
    -----------
    sim_df : pd.DataFrame
        DataFrame containing simulation data with columns ['battery_state', 'local_price'].
    save : bool, optional
        Whether to save the plot as an image file. Default is False.
    folder : str, optional
        Folder name inside 'results' where to save the plot. Default is 'simulation'.
    """
    # Categorize rows based on whether battery state of charge (SoC) is greater than zero
    sim_df["battery_used"] = sim_df["battery_state"].apply(
        lambda soc: "Battery Used" if soc > 0 else "No Battery"
        )

    plt.figure(figsize=(14, 6))

    # Boxplot comparing local prices when battery was used vs not used
    sns.boxplot(y="battery_used", x="local_price", data=sim_df)

    # Setting axis and title
    plt.title("Local Energy Price Distribution With vs. Without Battery Usage")
    plt.ylabel("Battery Usage")
    plt.xlabel("Local Energy Price (€/kWh)")
    plt.grid(True)

    # Save or show the plot
    if save:
        plt.savefig(
            f"results/{folder}/Impact_battery.png",
            dpi=300
            )
        plt.close()
    else:
        plt.show()


def energy_delta_per_agent(agent_df, save=False, folder="simulation"):
    """
    Plot the net energy balance (produced - consumed) per agent.

    Parameters:
    -----------
    agent_df : pd.DataFrame
        DataFrame with agent-level data containing 'agent_id', 'produced', 'consumed', and 'profile' columns.
    save : bool, optional
        Whether to save the plot as an image file. Default is False.
    folder : str, optional
        Folder name inside 'results' where to save the plot. Default is 'simulation'.
    """
    # Aggregate sum of produced and consumed energy by agent
    agent_summary = agent_df.groupby("agent_id").agg({
        "profile": "first",
        "produced": "sum",
        "consumed": "sum"
    }).reset_index()

    # Calculate net balance
    agent_summary["net_balance"] = agent_summary["produced"] \
        - agent_summary["consumed"]

    plt.figure(figsize=(12, 6))

    # Barplot sorted by net balance, colored by profile
    sns.barplot(
        data=agent_summary.sort_values("net_balance"),
        x="agent_id", y="net_balance", palette=PROFILE_COLORS,
        hue="profile"
        )
    
    # Setting axis and title
    plt.axhline(0, color='black', linestyle='--') # Line at zero for net balance reference
    plt.title("Net Energy Balance Per Agent (Produced - Consumed)")
    plt.xlabel("Agent ID")
    plt.ylabel("Net Energy Balance (kWh)")
    plt.tight_layout()

    # Save or show plot
    if save:
        plt.savefig(
            f"results/{folder}/Agent_net_energy.png",
            dpi=300
            )
        plt.close()
    else:
        plt.show()


def import_export_per_agent(agent_df, save=False, folder="simulation"):
    """
    Plot total imported and exported energy per agent, with background color indicating their profile.

    Parameters:
    -----------
    agent_df : pd.DataFrame
        DataFrame with agent-level data including 'agent_id', 'import', 'export', and 'profile' columns.
    save : bool, optional
        Whether to save the plot as an image file. Default is False.
    folder : str, optional
        Folder name inside 'results' where to save the plot. Default is 'simulation'.
    """
    # Extract one profile per agent
    agent_profiles = agent_df.groupby(
        'agent_id'
        )['profile'].first().reset_index()

    # Sum import and export per agent
    grouped = agent_df.groupby(
        'agent_id'
        )[['import', 'export']].sum().reset_index()

    # Merge profiles into grouped dataframe
    grouped = grouped.merge(agent_profiles, on='agent_id')

    # Setup bar plot parameters
    x = np.arange(len(grouped))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))

    # Add background color per agent profile
    for i, row in grouped.iterrows():
        color = PROFILE_COLORS.get(row['profile'], '#eeeeee')
        ax.axvspan(i - 0.4, i + 0.4, color=color, alpha=0.3)

    # Plot import and export bars
    ax.bar(
        x - width/2, grouped['import'],
        width, label='Import', color='red'
        )
    ax.bar(
        x + width/2, grouped['export'],
        width, label='Export', color='green'
        )

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['agent_id'])
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Totaal Traded Energy (kWh)')
    ax.set_title('Import en Export per Agent met Profielachtergrond')
    ax.legend()

    # Create legend patches for profiles
    legend_patches = [
        Patch(facecolor=color, edgecolor='none', label=profile)
        for profile, color
        in PROFILE_COLORS.items()
        ]

    # Add profile and import/export legends together
    ax.legend(
        handles=[
            *legend_patches,
            Patch(color='red', label='Import'),
            Patch(color='green', label='Export')
            ]
        )


    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save or show plot
    if save:
        plt.savefig(
            f"results/{folder}/Import_export.png",
            dpi=300
            )
        plt.close()
    else:
        plt.show()


def show_sim_report(sim, profile_ratios, save=False, folder="simulation"):
    """
    Generate and display or save a comprehensive text report summarizing simulation parameters and results.

    Parameters:
    -----------
    sim : MicroGrid
        The simulation instance containing results data.
    profile_ratios : dict
        Dictionary with profile names as keys and their ratios as values.
    save : bool, optional
        Whether to save the report to a file. Default is False.
    folder : str, optional
        Folder name inside 'results' where to save the report. Default is 'simulation'.
    """
    # Extract unique profiles from agent data
    unique_profiles = sim.agent_df['profile'].unique().tolist()

    # Aggregate mean energy statistics per profile
    agent_summary = sim.agent_df.groupby("profile")[
        [
            "produced", "consumed", "import",
            "export", "energy_centralgrid",
            "energy_microgrid"
        ]
    ].mean().reset_index()
    agent_summary["net_balance"] = agent_summary["produced"] - \
        agent_summary["consumed"]

    # Format profile ratio lines for report
    profile_ratio_lines = []
    for profile in unique_profiles:
        ratio = profile_ratios.get(profile, 'N/A')
        profile_ratio_lines.append(f"- {profile}: ratio = {ratio}")

    profile_ratio = "\n          ".join(profile_ratio_lines)

    # Prepare detailed profile report lines
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
            f"        - Avg Produced:     {produced:.4f} kW"
            )
        profile_report_lines.append(
            f"        - Avg Consumed:     {consumed:.4f} kW"
            )
        profile_report_lines.append(
            f"        - Avg Imported:     {import_:.4f} kW"
            )
        profile_report_lines.append(
            f"        - Avg Exported:     {export_:.4f} kW"
            )
        profile_report_lines.append(
            f"        - Avg From Grid:    {from_grid:.4f} kW"
            )
        profile_report_lines.append(
            f"        - Avg From Micro:   {from_microgrid:.4f} kW"
            )
        profile_report_lines.append(
            f"        - Net Balance:      {net_balance:.4f} kW"
            )

    profile_report = "\n            ".join(profile_report_lines)

    # Collect other simulation parameters and data
    total_demand = sim.sim_df["demand_from_centralgrid"].sum()
    total_supply = sim.sim_df["microgrid_supply"].sum()
    avg_energy_delta = sim.sim_df["energy_delta"].mean()
    avg_central_price = sim.sim_df["grid_price"].mean()
    min_central_price = sim.sim_df["grid_price"].min()
    max_central_price = sim.sim_df["grid_price"].max()
    avg_micro_price = sim.sim_df["local_price"].mean()
    min_micro_price = sim.sim_df["local_price"].min()
    max_micro_price = sim.sim_df["local_price"].max()
    std_micro_price = sim.sim_df["local_price"].std()
    avg_calc_prize = sim.sim_df["calculated_price"].mean()
    min_calc_prize = sim.sim_df["calculated_price"].min()
    max_calc_prize = sim.sim_df["calculated_price"].max()

    total_charge = sim.sim_df["battery_usage"].sum()
    avg_charge_perc = sim.sim_df["battery_state"].mean()
    max_charge = sim.sim_df["battery_state"].max()

    # Combine all sections into the final report string
    report_text = f"""
        Details of the simulation:

          Parameters:
          ---------------------------------------------------

          - Number of Households:   {sim.num_agents}
          - Gini:                   {sim.gini:.2f}
          - Mean panels:            {sim.mean_panels:.2f} panels
          - Panel efficiency:       {sim.panel_efficiency:.2f} (%)
          - Battery capacity:       {sim.bat_capacity:.2f} kWh
          - Battery charge rate:    {sim.bat_c_rate:.2f} (%)
          - Battery efficiency:     {sim.bat_efficiency:.2f} (%)
          - Number of days:         {len(sim.simulation_data)}

          Profiles used:
          ---------------------------------------------------
          {profile_ratio}

          Results:
          ---------------------------------------------------

            Energy statistics:
            -------------------------------------------------
            - Demand from central grid: {total_demand:.2f} kWh
            - Total supply:             {total_supply:.2f} kWh
            - Average energy delta:     {avg_energy_delta:.4f} kWh
            - Average grid price:       {avg_central_price:.4f} €/kWh
            - Min grid price:           {min_central_price:.4f} €/kWh
            - Max grid price:           {max_central_price:.4f} €/kWh
            - Average local price:      {avg_micro_price:.4f} €/kWh
            - Min local price:          {min_micro_price:.4f} €/kWh
            - Max local price:          {max_micro_price:.4f} €/kWh
            - Local price deviation:    {std_micro_price:.4f} €/kWh
            - Average calculated price: {avg_calc_prize:.4f} €/kWh
            - Min calculated price:     {min_calc_prize:.4f} €/kWh
            - Max calculated price:     {max_calc_prize:.4f} €/kWh

            Profiles:
            -------------------------------------------------
            {profile_report}

            Battery:
            -------------------------------------------------
            - Average State of Charge:  {avg_charge_perc:.2f} %
            - Maximum charge level:     {max_charge:.2f} %
            - Total battery coverage:   {total_charge:.2f} kWh

    """

    # Save or show text
    if save:
        with open(f"results/{folder}/simulation_report.txt", "w") as file:
            file.write(report_text)
    else:
        print(report_text)


def gather_results(params, folder):
    """
    Run the full microgrid simulation pipeline: generate data, run simulation,
    save results, create plots, and generate a report.

    Parameters:
    -----------
    params : dict
        Dictionary of simulation parameters. Expected keys include:
        - n_days (int): Number of simulation days.
        - n_households (int): Number of households in the simulation.
        - profile_ratios (dict): Ratios of different household profiles.
        - gini (float): Gini coefficient to model inequality in consumption.
        - mean_panels (float): Average number of solar panels per household.
        - panel_efficiency (float): Efficiency of solar panels.
        - battery_capacity (float): Battery capacity in kWh.
        - battery_charge_rate (float): Battery charging rate.
        - battery_efficiency (float): Battery round-trip efficiency.
        - seed (int, optional): Random seed for reproducibility.
    folder : str
        Folder name inside 'results' directory where all output files will be saved.

    Returns:
    --------
    None
    """
    # Ensure base 'results' directory exists
    results_path = "results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Create specific folder for this simulation run
    folder_path = os.path.join(results_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate consumption data for households based on profiles and days
    household_prosumption_data = generate_household_dataframe(
        n_days=params["n_days"],
        n_households=params["n_households"],
        profile_ratios=params["profile_ratios"],
        seed=42
        )

    # Load solar production data depending on simulation length
    if params["n_days"] > 366 and params["n_days"] < 732:
        production_data = pd.read_csv("data/solar_2years.csv", index_col="DATE")
    elif params["n_days"] > 0:
        production_data = pd.read_csv("data/solar_strength.csv", index_col="DATE")
    else:
        # Currently only supports up to two years
        return print(
            "Currently the code only supports up to two years."
            "Please adjust the `n_days` parameter accodingly!"
            )

    # Generate grid price data based on number of days and solar dataset used
    if params["n_days"] > 366 and params["n_days"] < 732:
        grid_price_data = generate_grid_prize_data(
            n_days=params["n_days"],
            solar_csv_path="data/solar_2years.csv",
            seed=42
            )
    elif params["n_days"] > 0:
        grid_price_data = generate_grid_prize_data(
            n_days=params["n_days"],
            seed=42
            )

    # Initialize MicroGrid simulation object with all parameters and data
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

    # Run simulation for the total hours (days - 1) * 24
    simulation.long_step(
        n=(params["n_days"] - 1) * 24
        )

    # Convert simulation results to DataFrame for analysis and plotting
    simulation._convert_to_dataframe()

    # Save simulation results dataframes as CSV files
    simulation.sim_df.to_csv(
        f"results/{folder}/simulation_data.csv", sep=";"
    )

    simulation.agent_df.to_csv(
        f"results/{folder}/agent_data.csv", sep=";"
    )

    # Generate and save plots to results folder
    grid_price_vs_local_price(simulation.sim_df, save=True, folder=folder)
    impact_of_battery_usage(simulation.sim_df, save=True, folder=folder)
    energy_delta_per_agent(simulation.agent_df, save=True, folder=folder)
    import_export_per_agent(simulation.agent_df, save=True, folder=folder)

    # Generate and save simulation textual report
    show_sim_report(
        simulation, params["profile_ratios"],
        save=True, folder=folder
        )

    # Inform user of completion and results location
    print(f"""
    Results have been gathered!

    You can find the gathered results in the folder results/{folder}
    """)
