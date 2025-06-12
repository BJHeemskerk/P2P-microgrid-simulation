import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def grid_prize_vs_local_prize(sim_df):
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
    plt.show()


def impact_of_battery_usage(sim_df):
    sim_df["battery_used"] = sim_df["battery_state"].apply(
        lambda soc: "Battery Used" if soc > 0 else "No Battery"
        )

    plt.figure(figsize=(14, 6))
    sns.boxplot(y="battery_used", x="local_price", data=sim_df)
    plt.title("Local Energy Price Distribution With vs. Without Battery Usage")
    plt.ylabel("Battery Usage")
    plt.xlabel("Local Energy Price (€/kWh)")
    plt.grid(True)
    plt.show()


def energy_delta_per_agent(agent_df):
    agent_summary = agent_df.groupby("agent_id")[
        ["profile", "produced", "consumed"]
        ].sum().reset_index()
    agent_summary["net_balance"] = agent_summary["produced"] \
        - agent_summary["consumed"]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=agent_summary.sort_values("net_balance"),
        x="agent_id", y="net_balance", palette="viridis",
        hue="profile", legend=False
        )
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Net Energy Balance Per Agent (Produced - Consumed)")
    plt.xlabel("Agent ID")
    plt.ylabel("Net Energy Balance (kWh)")
    plt.tight_layout()
    plt.show()


def show_sim_report(sim, profile_ratios):
    unique_profiles = sim.agent_df['profile'].unique().tolist()

    profile_report_lines = []
    for profile in unique_profiles:
        ratio = profile_ratios.get(profile, 'N/A')
        profile_report_lines.append(f"- {profile}: ratio = {ratio}")

    profile_report = "\n          ".join(profile_report_lines)

    print(f"""
        Details of the simulation:

          Parameters:
          ---------------------------------------------------

          - Number of Households:   {sim.num_agents}
          - Gini:                   {sim.gini}
          - Mean panels:            {sim.mean_panels}
          - Panel efficiency:       {sim.panel_efficiency}
          - Battery capacity:       {sim.bat_capacity}
          - Battery charge rate:    {sim.bat_c_rate}
          - Battery efficiency:     {sim.bat_efficiency}
          - Number of days:         {len(sim.simulation_data) / 24}


          Profiles used:
          ---------------------------------------------------
          {profile_report}


          Results:
          ---------------------------------------------------

            Energy statistics:
            -------------------------------------------------
            - Total demand:             {sim.sim_df["microgrid_demand"].sum()}
            - Total supply:             {sim.sim_df["microgrid_supply"].sum()}
            - Average energy delta:     {sim.sim_df["energy_delta"].mean()}
            - Average grid price:       {sim.sim_df["grid_price"].mean()}
            - Min grid price:           {sim.sim_df["grid_price"].min()}
            - Max grid price:           {sim.sim_df["grid_price"].max()}
            - Average local price:      {sim.sim_df["local_price"].mean()}
            - Min local price:          {sim.sim_df["local_price"].min()}
            - Max local price:          {sim.sim_df["local_price"].max()}
            - Local price deviation:    {sim.sim_df["local_price"].std()}

""")
