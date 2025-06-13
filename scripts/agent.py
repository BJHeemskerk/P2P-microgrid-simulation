import mesa


class Household(mesa.Agent):
    """
    The Household Agent is the basic Agent template used during our
    simulations. It is able to withdraw information about its own consumption
    and production details from our synthetically generated data. Next to that,
    it holds the capacity to trade energy with other Agents in the simulation.

    Methods:
    ----------
    trade_energy(self)
        A method that functions as the main use for our Agents. Using this
        method makes the agent look up the needed consumption and production
        data, along with trading energy with other Agents.

    _calculate_energy(self)
        A method used to calculate the energy surplus or deficit, using its
        corresponding consumption and solar strength values. The producyion
        is calculated as follows:
            Production = solar strength * (solar panels * area per solar panel)

    _lookup_data_for_agent(self, data)
        A method that uses the dataframe in order to find the needed
        consumption amount.

    _calculate_production_for_agent(self, data)
        A method that uses the dataframe to find the solar strength and
        calculates the energy production.

    _trading(self)
        A method holding the trading methods and logic for when not to
        start trading.

    _look_for_trades(self)
        A method that looks through the Agents for potential buyers
        and sellers.

    _exchange_energy(self)
        A method that handles the logic to exchange energy until either
        the buyer has enough or until the seller no longer has a surplus.
    """

    def __init__(self, model):
        """
        Initializes a household agent with a random number of solar panels
        and several other attributes needed to run the code without errors.
        """
        super().__init__(model)

        self.solar_panel_area = round(
            self.solar_panels_at_position(
                Gini=self.model.gini,
                mean_panels=self.model.mean_panels,
                n=30,
                i=self.unique_id
            ), 1)

        self.supply = 1
        self.demand = 1

        self.remaining_energy = 0
        self.consumed = 0
        self.produced = 0
        self.energy_from_microgrid = 0
        self.energy_from_centralgrid = 0
        self.imported_energy = 0
        self.exported_energy = 0
        self.traded_energy = 0

    def trade_energy(self):
        self._calculate_energy()
        self._trading()

        if self.remaining_energy < 0:
            self.energy_from_centralgrid = abs(self.remaining_energy)
            self.model.hourly_demand += self.energy_from_centralgrid
            self.remaining_energy = 0
        else:
            self.model.hourly_supply += self.remaining_energy

        self._update_dataframe()

        # Resetting data parameters
        self.traded_energy = 0
        self.energy_from_microgrid = 0
        self.energy_from_centralgrid = 0
        self.imported_energy = 0
        self.exported_energy = 0

    def solar_panels_distribution(self, Gini, mean_panels, n):
        k = Gini / (1 - Gini)

        # Calculating the raw division of panels
        raw = [(2 * (i + 0.5) / n) ** k for i in range(n)]

        # Normalising the panels to ensure mean_panels * n
        total_raw = sum(raw)
        factor = (mean_panels * n) / total_raw
        normalized = [x * factor for x in raw]

        return normalized

    def solar_panels_at_position(self, Gini, mean_panels, n, i):
        return self.solar_panels_distribution(Gini, mean_panels, n)[i - 1]

    def _calculate_energy(self):
        # Determine the consumption amount at the current hour
        self.consumed = self._lookup_consumption_for_agent(
            self.model.consumption_data
        )

        # Determine and calculate the production at the current hour
        self.produced = self._calculate_production_for_agent(
            self.model.production_data
        )

        # Calculate the remaining energy after production and consumption
        self.remaining_energy = self.produced - self.consumed

    def _lookup_consumption_for_agent(self, data):
        # Identify the agent and the current day
        household_id = f"household_{self.unique_id}"
        day_data = data.loc[self.model.day_str]

        # Look up consumption value for the agent on this day and hour
        lookup_data = day_data[
            day_data["household"] == household_id
        ][f"{self.model.hour}"].values[0]

        return lookup_data

    def _calculate_production_for_agent(self, data):
        # Determine the solar strength at the current hour
        solar_strength = data.loc[
            self.model.day_str, f"{self.model.hour}"
        ]

        # Wh = W/m2 * m2 * time * panel_effiency
        # kWh = Wh / 1000
        # The time for us is already per hour, hence the value of 1
        produced = (
            solar_strength
            * self.solar_panel_area
            * 1
            * self.model.panel_efficiency
        ) / 1000

        return produced

    def _trading(self):
        self._look_for_trades()

        if not self.buyers:
            return

        if self.remaining_energy <= 0:
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

    def _exchange_energy(self, buyers):
        for buyer in buyers:
            # Calculating the amount to be traded
            potential_trade = buyer.consumed - buyer.remaining_energy
            if potential_trade <= 0:
                continue

            # Setting the trade amount to cap out at surplus of seller
            trade_amount = min(self.remaining_energy, potential_trade)

            if trade_amount <= 0:
                continue

            # Exchanging the energy
            self.remaining_energy -= trade_amount
            buyer.remaining_energy += trade_amount

            self.exported_energy += trade_amount
            buyer.energy_from_microgrid += trade_amount
            buyer.imported_energy += trade_amount

            if self.remaining_energy <= 0:
                break

    def _update_dataframe(self):
        self.profile = self.model.consumption_data[
            self.model.consumption_data["household"]
            == f"household_{self.unique_id}"
        ]["profiel"].values[0]

        agent_data = {
            "profile": self.profile,
            "solarpanels": self.solar_panel_area,
            "consumed": self.consumed,
            "produced": self.produced,
            "energy_microgrid": self.energy_from_microgrid,
            "energy_centralgrid": self.energy_from_centralgrid,
            "import": self.imported_energy,
            "export": self.exported_energy
        }

        day_data = self.model.agent_data[self.model.day_str]
        day_data[self.model.hour][self.unique_id] = agent_data
