import mesa
import numpy as np
from datetime import datetime
from utils import Battery
from agent import Household
from dateutil.relativedelta import relativedelta
from collections import defaultdict


class MicroGrid(mesa.Model):
    def __init__(self, n, consumption_data, production_data, grid_prize_data, solar_panel_odds=0.35, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.solar_panel_odds = solar_panel_odds

        self.consumption_data = consumption_data
        self.production_data = production_data
        self.grid_price_df = grid_prize_data

        self.datetime = datetime(2024, 1, 1)
        self.hour = 0
        self.day_str = self.datetime.strftime("%d-%m-%Y")

        self.hourly_demand = [0] * 24
        self.hourly_supply = [0] * 24
        self.energy_delta = 0

        self.grid_price = self.grid_price_df.loc[self.day_str].values[0]
        self.energy_price = self.grid_price - 0.01

        self.smoothing = 0.3
        self.elasticity = 0.4
        self.min_price = 0.05

        self.simulation_data = defaultdict(dict)
        self.agent_data = defaultdict(lambda: defaultdict(dict))

        Household.create_agents(model=self, n=n)
        self.grid_battery = Battery(500, 0.5, 0.9)

    def _time_skip(self):
        if self.hour != 23:
            self.hour += 1
        else:
            self.datetime += relativedelta(days=1)
            self.day_str = self.datetime.strftime("%d-%m-%Y")
            self.hour = 0

            self._update_energy_price()

            # Reset hourly demand/supply for next day
            self.hourly_demand = [0] * 24
            self.hourly_supply = [0] * 24

    def _handle_battery_charge(self):
        net_production = sum(agent.produced for agent in self.agents)
        net_consumption = sum(agent.consumed for agent in self.agents)
        net_deficit = net_consumption - net_production

        if net_deficit > 0:
            self._handle_battery_deficit()
        else:
            self._handle_battery_surplus()

        if not hasattr(self, "battery_soc_history"):
            self.battery_soc_history = []
        self.battery_soc_history.append(self.grid_battery.state_of_charge)

    def _handle_battery_deficit(self, net_deficit):
        # Determine how much battery can cover
        battery_coverage = min(
            net_deficit,
            self.grid_battery.max_discharge,
            self.grid_battery.state_of_charge
            )

        # Discharge the battery
        if battery_coverage > 0:
            self.grid_battery.discharge(power_kw=battery_coverage, duration_hr=1)

        # Now `net_deficit` is the amount still needed from the external grid
        self.hourly_supply[self.hour] += battery_coverage

    def _handle_battery_surplus(self, net_deficit):
        net_surplus = -1 * net_deficit
        self.grid_battery.charge(power_kw=net_surplus, duration_hr=1)

    def _update_energy_price(self):
        # Grid price update
        self.grid_price = self.grid_price_df.loc[self.day_str].values[0]

        # Calculating target price
        target_price = self._calculate_target_price()

        # --- Smoothed Transition ---
        self.energy_price = (1 - self.smoothing) * self.energy_price + self.smoothing * target_price

        # Ensuring it never exceeds the grid price
        self.energy_price = max(self.min_price, min(self.energy_price, self.grid_price))

        # Output energy price for the day
        print(f"[DAILY PRICE UPDATE :: {self.day_str}]: "
            f"Price={self.energy_price:.4f} | "
            f"Grid Price={self.grid_price:.4f} | "
            # f"Supply={self.hourly_supply}\n"
            # f"Demand={self.hourly_demand}"
            )

    def _calculate_target_price(self):
        demand_list = self.hourly_demand
        supply_list = self.hourly_supply

        # Avoid zero division
        valid_hours = [i for i in range(24) if supply_list[i] > 0]
        if not valid_hours:
            return self.grid_price  # fallback

        hourly_pressures = [
        demand_list[i] / supply_list[i] for i in valid_hours
        ]
        
        # Calculate the average pressure across valid hours
        self.market_pressure = np.mean(hourly_pressures)

        # If demand exceeds supply, increase the price, but with diminishing returns
        if self.market_pressure > 1:
            # The more extreme the imbalance, the stronger the price increase
            price_increase = (self.market_pressure - 1) * self.elasticity
            new_price = self.energy_price * (1 + price_increase)
        else:
            # If supply exceeds demand, decrease the price but not too sharply
            price_decrease = (1 - self.market_pressure) * self.elasticity
            new_price = self.energy_price * (1 - price_decrease)

        return new_price

    def collect_hourly_data(self):
        microgrid_demand = self.hourly_demand[self.hour]
        microgrid_supply = self.hourly_supply[self.hour]

        sim_data = {
            "grid_price": self.grid_price,
            "local_price": self.energy_price,
            "market_pressure": microgrid_demand / microgrid_supply if microgrid_supply > 0 else np.nan,
            "microgrid_demand": microgrid_demand,
            "microgrid_supply": microgrid_supply,
            "energy_delta": microgrid_supply - microgrid_demand
        }

        self.simulation_data[f"{self.day_str}"][f"{self.hour}"] = sim_data

    def step(self):
        self.agents.shuffle_do("trade_energy")
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
        for _ in range(0, n):
            self.agents.shuffle_do("trade_energy")

            # Collect data for the day
            self.collect_hourly_data()
            self._time_skip()
