import mesa
import numpy as np
import pandas as pd
from datetime import datetime
from scripts.utils import Battery
from scripts.agent import Household
from dateutil.relativedelta import relativedelta
from collections import defaultdict


class MicroGrid(mesa.Model):
    """
    MicroGrid model for simulating a decentralized energy market between households.
    
    Each household can consume and produce electricity, trade energy locally,
    and interact with a central battery or the main energy grid.
    
    Attributes:
        agents: List of all Household agents
        grid_battery: Shared battery object for the entire microgrid
        simulation_data: Dictionary storing system-wide stats by time
        agent_data: Dictionary storing individual household stats by time
    """
    
    def __init__(
        self, n_households, consumption_data, production_data,
        grid_prize_data, gini=0.1,
        mean_panels=8, panel_efficiency=0.2, bat_capacity=500,
        bat_c_rate=0.5, bat_efficiency=0.9, seed=None,
        verbose=1
    ):
        """
        Initialize the microgrid simulation.

        Args:
            n_households (int): Number of household agents in the model.
            consumption_data (DataFrame): Hourly consumption data for each household.
            production_data (DataFrame): Hourly solar strength data.
            grid_prize_data (DataFrame): Grid electricity prices by hour.
            gini (float): Gini coefficient for inequality in solar panel distribution.
            mean_panels (int): Mean number of panels per household.
            panel_efficiency (float): Efficiency of the solar panels.
            bat_capacity (float): Capacity of the microgrid battery (kWh).
            bat_c_rate (float): Max charge/discharge rate as a fraction of capacity.
            bat_efficiency (float): Round-trip efficiency of the battery.
            seed (int, optional): Random seed for reproducibility.
            verbose (int): Level of verbosity for logging output.
        """
        super().__init__(seed=seed)
        self.verbose = verbose
        self.num_agents = n_households

        # Input data
        self.consumption_data = consumption_data
        self.production_data = production_data
        self.grid_price_df = grid_prize_data

        # Energy production parameters
        self.gini = gini
        self.mean_panels = mean_panels
        self.panel_efficiency = panel_efficiency

        # Battery setup
        self.bat_capacity = bat_capacity
        self.bat_c_rate = bat_c_rate
        self.bat_efficiency = bat_efficiency
        self.grid_battery = Battery(bat_capacity, bat_c_rate, bat_efficiency)

        # Simulation time tracking
        self.datetime = datetime(2024, 1, 1)
        self.hour = 0
        self.day_str = self.datetime.strftime("%d-%m-%Y")

        # Energy metrics
        self.hourly_demand = 0
        self.hourly_supply = 0
        self.energy_delta = 0
        self.energy_from_battery = 0

        # Pricing logic
        self.grid_price = self.grid_price_df.loc[self.day_str].values[0]
        self.energy_price = self.grid_price - 0.01  # Slightly cheaper than grid
        self.calculated_price = 0
        self.smoothing = 0.3  # How quickly local price reacts to demand/supply
        self.elasticity = 0.4  # How sensitive the price is to market pressure
        self.min_price = 0.05  # Minimum floor price

        # Data recording
        self.simulation_data = defaultdict(dict)
        self.agent_data = defaultdict(lambda: defaultdict(dict))

        # Create household agents
        Household.create_agents(model=self, n=n_households)

    def _time_skip(self):
        """
        Move forward in simulation time by one hour.
        Resets hourly counters and handles daily transitions.
        """
        # If the hour is not 23, move to the next hour
        if self.hour != 23:
            self.hour += 1
        else:
            # Move to the next day
            self.datetime += relativedelta(days=1)
            self.day_str = self.datetime.strftime("%d-%m-%Y")
            self.hour = 0

            # Print a short update
            if self.verbose > 0:
                print(
                    f"[DAILY PRICE UPDATE :: {self.day_str}]: "
                    f"Price: {self.energy_price:.4f} | "
                    f"Grid Price: {self.grid_price:.4f} | "
                    f"Battery charge level: {self.grid_battery.soc_percent:.2f}"
                )

        # Update the energy prices (central- and microgrid)
        self._update_energy_price()

        # Reset hourly stats
        self.energy_from_battery = 0
        self.hourly_demand = 0
        self.hourly_supply = 0

    def _handle_battery_charge(self):
        """
        Manage battery interactions depending on net energy balance.
        Charges battery on surplus; discharges on deficit.
        """
        # Calculate the deficit, if any
        net_production = sum(agent.produced for agent in self.agents)
        net_consumption = sum(agent.consumed for agent in self.agents)
        net_deficit = net_consumption - net_production

        # If there is a deficit, handle it through the battery
        if net_deficit > 0:
            discharged = self._handle_battery_deficit(net_deficit)
            self._distribute_battery_energy(discharged)
        else:
            # On a surplus, add energy to the battery
            self._handle_battery_surplus(net_deficit)

    def _handle_battery_deficit(self, net_deficit):
        """
        Discharge battery to cover a supply deficit.
        """
        # Find available possible coverage from the battery
        battery_coverage = min(
            net_deficit,
            self.grid_battery.max_discharge,
            self.grid_battery.state_of_charge
        )

        # If the coverage is higher than 0, discharge energy
        if battery_coverage > 0:
            self.grid_battery.discharge(power_kw=battery_coverage, duration_hr=1)

        # Update statistics for price calculation and tracking
        self.energy_from_battery = battery_coverage
        self.hourly_supply += battery_coverage
        self.hourly_demand -= battery_coverage

        return battery_coverage

    def _distribute_battery_energy(self, battery_coverage):
        """
        Distribute battery energy to households still needing energy.
        """
        # Look for 'needy agents'
        needy_agents = [
            agent
            for agent
            in self.agents
            if agent.remaining_energy < 0
            ]
        
        # Calculate total deficit
        total_deficit = sum(abs(agent.remaining_energy) for agent in needy_agents)

        # Stop if unable to distribute energy
        if total_deficit == 0 or battery_coverage == 0:
            return

        # Distribute the energy to the agents
        for agent in needy_agents:
            # Determining the share per agent
            agent_deficit = abs(agent.remaining_energy)
            share_ratio = agent_deficit / total_deficit
            battery_share = min(battery_coverage * share_ratio, agent_deficit)

            agent.energy_from_microgrid += battery_share
            agent.energy_from_centralgrid = max(
                agent.energy_from_centralgrid - battery_share, 0
                )
            agent.remaining_energy += battery_share

    def _handle_battery_surplus(self, net_deficit):
        """
        Charge the battery with excess local production.
        """
        # Calculate the amount of surplus
        net_surplus = -1 * net_deficit

        # Charge the battery
        self.grid_battery.charge(power_kw=net_surplus, duration_hr=1)

    def _update_energy_price(self):
        """
        Update the local microgrid energy price using market pressure and smoothing.
        """
        # Lookup grid price
        self.grid_price = self.grid_price_df.loc[self.day_str][f"{self.hour}"]

        # Calculate new target price
        self.calculated_price = self._calculate_target_price()

        # Apply exponential smoothing
        self.energy_price = (
            (1 - self.smoothing)
            * self.energy_price
            + self.smoothing
            * self.calculated_price
        )

        # Keep within valid bounds
        self.energy_price = max(
            self.min_price, min(self.energy_price, self.grid_price)
            )

    def _calculate_target_price(self):
        """
        Calculate the theoretical energy price based on supply and demand.
        """
        # If supply is 0, keep price at grid level
        if self.hourly_supply == 0:
            return self.grid_price

        # Calculate the average pressure across valid hours
        self.market_pressure = self.hourly_demand / self.hourly_supply

        if self.market_pressure > 1:
            # If demand exceeds supply, increase the price but not too sharp
            price_increase = (
                (self.market_pressure - 1) / (1 + self.market_pressure)
                ) * self.elasticity
            new_price = self.energy_price * (1 + price_increase)
        else:
            # If supply exceeds demand, decrease the price but not too sharp
            price_decrease = (
                (1 - self.market_pressure) / (1 + self.market_pressure)
                ) * self.elasticity
            new_price = self.energy_price * (1 - price_decrease)

        return new_price

    def collect_hourly_data(self):
        """
        Record simulation-level statistics for the current hour.
        """
        # Gather the simulation data
        sim_data = {
            "grid_price": self.grid_price,
            "local_price": self.energy_price,
            "calculated_price": self.calculated_price,
            "market_pressure": self.market_pressure,
            "demand_from_centralgrid": self.hourly_demand,
            "microgrid_supply": self.hourly_supply,
            "energy_delta": self.hourly_supply - self.hourly_demand,
            "battery_usage": self.energy_from_battery,
            "battery_state": self.grid_battery.soc_percent
        }

        # Insert into dictionary
        self.simulation_data[self.day_str][f"{self.hour}"] = sim_data

    def _convert_to_dataframe(self):
        """
        Convert the stored simulation and agent data to Pandas DataFrames.
        """
        # Convert system-wide data
        records = []
        for day, hours in self.simulation_data.items():
            for hour, data in hours.items():
                record = {"day": day, "hour": int(hour)}
                record.update(data)
                records.append(record)

        self.sim_df = pd.DataFrame(records)

        # Convert agent-specific data
        agent_records = []
        for day, hours in self.agent_data.items():
            for hour, agents in hours.items():
                for agent_id, data in agents.items():
                    record = {
                        "day": day,
                        "hour": int(hour),
                        "agent_id": agent_id
                    }
                    record.update(data)
                    agent_records.append(record)

        self.agent_df = pd.DataFrame(agent_records)

    def step(self):
        """
        Execute one hour of simulation, including energy trading for all agents.
        Logs agent stats if verbosity is enabled.
        """
        # Trade energy
        self.agents.shuffle_do("trade_energy")

        # Print small report for all agents
        for agent in self.agents:
            print(f"""
            Statistics for Agent {agent}:
            Current day: {self.day_str}
            Current energy price: {self.energy_price}
            Current remaining energy: {agent.remaining_energy}
            Last consumed: {agent.consumed}
            Last produced: {agent.produced}
            Total agent amount: {self.num_agents}
            """)

    def long_step(self, n):
        """
        Run the simulation for `n` consecutive hours.

        Args:
            n (int): Number of hours to simulate.
        """
        for _ in range(n):
            # Trade energy
            self.agents.shuffle_do("trade_energy")

            # Use battery if needed
            self._handle_battery_charge()

            # Collect data
            self.collect_hourly_data()
            
            # Go to the next step
            self._time_skip()
