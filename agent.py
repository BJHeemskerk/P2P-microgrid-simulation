import mesa


class Household(mesa.Agent):
    """
    The Household Agent is the basic Agent template used during our sumulations.
    It is able to withdraw information about its own consumption and production details
    from our synthetically generated data. Next to that, it holds the capacity to trade
    energy with other Agents in the simulation.

    Methods:
    ----------
    trade_energy(self)
        A method that functions as the main use for our Agents. Using this method makes
        the agent look up the needed consumption and production data, along with trading
        energy with other Agents.

    _calculate_energy(self)
        A method used to calculate the energy surplus or deficit, using its corresponding
        consumption and solar strength values. The producyion is calculated as follows:
            Production = solar_strength * (amount_of_solar_panels * avg_area_per_solar_panel)
    
    _lookup_data_for_agent(self, data)
        A method that uses the dataframe in order to find the needed consumption and
        solar strength value.
    
    _trading(self)
        A method holding the trading methods and logic for when not to start trading.
    
    _look_for_trades(self)
        A method that looks through the Agents for potential buyers and sellers.

    _exchange_energy(self)
        A method that handles the logic to exchange energy until either the buyer has
        enough or until the seller no longer has a surplus.
    """

    def __init__(self, model):
        """
        Initializes a household agent with a random number of solar panels and several other
        attributes needed to run the code without errors.
        """
        super().__init__(model)

        self.solar_panel_area = round(self.solar_panels_at_position(Gini=0.1, mean_panels=8, n=30, i=self.unique_id), 1)

        self.supply = 1
        self.demand = 1

        self.remaining_energy = 0
        self.traded_energy = 0
        self.consumed = 0
        self.produced = 0
        self.trade_amount = 0

        # print(f"[INIT] Household initialized with {self.amount_of_solarpanels} solar panels.")
        # print(f"[INIT] Consumption: {self.consumption_data} | Production: {self.production_data}")

    def trade_energy(self):
        # print(f"\n[TRADE] Time: Day {self.model.day_str}, Hour {self.model.hour}")
        self._calculate_energy()
        self._trading()

        if self.remaining_energy > 0:
            self.model.hourly_supply[self.model.hour] += self.remaining_energy
        else:
            self.model.hourly_demand[self.model.hour] += abs(self.remaining_energy)

        self._update_dataframe()
        self.traded_energy = 0

    def solar_panels_distribution(self, Gini, mean_panels, n):
        k = Gini / (1 - Gini)
    
        # Bereken ruwe verdeling
        raw = [(2 * (i + 0.5) / n) ** k for i in range(n)]
        
        # Normaliseer zodat het totaal gelijk blijft aan mean_panels * n
        total_raw = sum(raw)
        factor = (mean_panels * n) / total_raw
        normalized = [x * factor for x in raw]
    
        return normalized

    def solar_panels_at_position(self, Gini, mean_panels, n, i):
        return self.solar_panels_distribution(Gini, mean_panels, n)[i - 1]

    def _calculate_energy(self):
        self.consumed = self._lookup_data_for_agent(self.model.consumption_data)
        self.solar_strength = self.model.production_data.loc[self.model.day_str, f"{self.model.hour}"]

        # Wh = (W/m2 * m2) * h
        # kWh = Wh / 1000
        self.produced = (self.solar_strength * self.solar_panel_area * 0.2) / 1000

        self.remaining_energy = self.produced - self.consumed
        # print(f"[CALC] Consumed: {self.consumed}, Produced: {self.produced}, Remaining: {self.remaining_energy}")

    def _lookup_data_for_agent(self, data):
        household_id = f"household_{self.unique_id}"
        day_data = data.loc[self.model.day_str]
        lookup_data = day_data[day_data["household"] == household_id][f"{self.model.hour}"].values[0]
        return lookup_data

    def _trading(self):
        self._look_for_trades()
        if not self.buyers:
            # print("[TRADE] No buyers found.")
            return

        if self.remaining_energy <= 0:
            # print("[TRADE] No surplus energy to trade.")
            return

        self._exchange_energy(self.buyers)

    def _look_for_trades(self):
        self.buyers = [
            agent for agent in self.model.agents
            if agent.remaining_energy < agent.consumed
        ]
        self.sellers = [
            agent for agent in self.model.agents
            if agent.remaining_energy > agent.consumed
        ]
        # print(f"[MARKET] Buyers: {len(self.buyers)} | Sellers: {len(self.sellers)}")

    def _exchange_energy(self, buyers):
        for buyer in buyers:
            potential_trade = buyer.consumed - buyer.remaining_energy
            if potential_trade <= 0:
                continue

            trade_amount = min(self.remaining_energy, potential_trade)

            if trade_amount <= 0:
                continue

            self.remaining_energy -= trade_amount
            buyer.remaining_energy += trade_amount

            self.traded_energy += trade_amount
            buyer.traded_energy -= trade_amount

            # print(f"[EXCHANGE] Traded {trade_amount} units to Buyer (ID: {buyer.unique_id})")
            # print(f"[TRADER] Remaining energy: {self.remaining_energy}W")

            if self.remaining_energy <= 0:
                # print("[TRADE] No more energy to trade.")
                break

    def _update_dataframe(self):
        earnings = self.traded_energy * (self.model.grid_price - self.model.energy_price)

        agent_data = {
            "solarpanel_area": self.solar_panel_area,
            "consumed": self.consumed,
            "produced": self.produced,
            "traded": self.traded_energy,
            "earnings": earnings
        }

        self.model.agent_data[f"{self.model.day_str}"][f"{self.model.hour}"][f"{self.unique_id}"] = agent_data
