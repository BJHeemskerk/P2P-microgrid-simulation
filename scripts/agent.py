import mesa


class Household(mesa.Agent):
    """
    A Household agent in the energy trading simulation.

    Each agent represents a household capable of producing energy via solar panels,
    consuming energy, and trading surplus energy with other agents (households).
    
    The agent retrieves synthetic consumption and production data per time step
    and makes trading decisions based on its energy balance.

    Attributes:
    -----------
    solar_panel_area : float
        The effective area of solar panels assigned to the household.
    consumed : float
        Amount of energy consumed during the current time step.
    produced : float
        Amount of energy produced during the current time step.
    remaining_energy : float
        Net energy available (production - consumption).
    traded_energy : float
        Total amount of energy traded with other agents.
    energy_from_microgrid : float
        Energy imported via peer-to-peer trading.
    energy_from_centralgrid : float
        Energy drawn from the central grid (if trading wasn’t sufficient).
    imported_energy : float
        Total energy received from other agents.
    exported_energy : float
        Total energy given to other agents.
    """

    def __init__(self, model):
        """
        Initialize the Household agent.

        Randomly assigns a solar panel area based on the agent's position in the
        distribution and initializes all relevant tracking attributes.
        """
        super().__init__(model)
        # Assign solar panel area using a Gini-distributed method
        self.solar_panel_area = round(
            self.solar_panels_at_position(
                Gini=self.model.gini,
                mean_panels=self.model.mean_panels,
                n=30,
                i=self.unique_id
            ), 1)

        # Initialize default values for energy consumption and production tracking
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
        """
        Main method called each simulation step to perform energy-related actions.

        Steps:
            1. Calculate current energy consumption and production.
            2. Attempt to trade energy with other agents.
            3. If there is still an energy deficit, draw from the central grid.
            4. Update the shared data store with agent metrics.
        """
        # Calculate the current supply and demand and start trading
        self._calculate_energy()
        self._trading()

        if self.remaining_energy < 0:
            # If not enough energy, draw from the central grid
            self.energy_from_centralgrid = abs(self.remaining_energy)
            self.model.hourly_demand += self.energy_from_centralgrid
            self.remaining_energy = 0
        else:
            # Surplus energy contributes to grid-level supply
            self.model.hourly_supply += self.remaining_energy

        self._update_dataframe()

        # Reset metrics after each timestep
        self.traded_energy = 0
        self.energy_from_microgrid = 0
        self.energy_from_centralgrid = 0
        self.imported_energy = 0
        self.exported_energy = 0

    def solar_panels_distribution(self, Gini, mean_panels, n):
        """
        Creates a synthetic distribution of solar panels across all agents.

        Uses a function of the Gini coefficient to simulate inequality in solar
        panel ownership.

        Returns:
            List[float]: Normalized solar panel count per agent.
        """
        k = Gini / (1 - Gini)

        # Calculate the raw division of panels
        raw = [(2 * (i + 0.5) / n) ** k for i in range(n)]

        # Normalize to ensure the average panel count is preserved
        total_raw = sum(raw)
        factor = (mean_panels * n) / total_raw
        normalized = [x * factor for x in raw]

        return normalized

    def solar_panels_at_position(self, Gini, mean_panels, n, i):
        """
        Retrieves the number of solar panels for a specific agent position.

        Returns:
            float: The assigned solar panel area for the agent.
        """
        return self.solar_panels_distribution(Gini, mean_panels, n)[i - 1]

    def _calculate_energy(self):
        """
        Computes the agent's energy balance for the current timestep.

        Retrieves consumption and production values and updates remaining energy.
        """
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
        """
        Retrieves the hourly consumption value for this agent.

        Args:
            data (pd.DataFrame): The synthetic consumption data.

        Returns:
            float: Consumption value for the current hour.
        """
        # Identify the agent and the current date
        household_id = f"household_{self.unique_id}"
        day_data = data.loc[self.model.day_str]

        # Look up the consumption value in the dataframe for this hour
        return day_data[
            day_data["household"] == household_id
        ][f"{self.model.hour}"].values[0]

    def _calculate_production_for_agent(self, data):
        """
        Calculates energy production based on solar strength and panel area.

        Args:
            data (pd.DataFrame): The synthetic solar strength data.

        Returns:
            float: Energy produced in kWh.
        """
        # Determine the solar strength at the current hour
        solar_strength = data.loc[
            self.model.day_str, f"{self.model.hour}"
        ]

        # Wh = W/m2 * m2 * time (in h) * panel_effiency
        produced = (
            solar_strength
            * self.solar_panel_area
            * 1  # 1 hour duration
            * self.model.panel_efficiency
        ) / 1000  # Convertion to kWh

        return produced

    def _trading(self):
        """
        Main logic for trading with other agents.

        Skips trading if the agent has no energy surplus or no buyers are available.
        """
        # Start looking for trades
        self._look_for_trades()

        if not self.buyers:
            return

        if self.remaining_energy <= 0:
            return

        # Exchange energy with other agents
        self._exchange_energy(self.buyers)

    def _look_for_trades(self):
        """
        Finds agents who want to buy or sell energy.

        Populates self.buyers for use in trading logic.
        """
        # Look for buyers
        self.buyers = [
            agent for agent in self.model.agents
            if agent.remaining_energy < agent.consumed
        ]

    def _exchange_energy(self, buyers):
        """
        Transfers energy from this agent (seller) to buyers with deficits.

        The exchange continues until this agent runs out of surplus or all buyers
        meet their demand.

        Args:
            buyers (List[Household]): List of buyer agents needing energy.
        """
        # Loop over the buyers
        for buyer in buyers:
            # Calculate the amount of energy the buyer wants to trade
            potential_trade = buyer.consumed - buyer.remaining_energy

            # If the potential trade is 0 or less, cancel trade attempt
            if potential_trade <= 0:
                continue

            # Set bounds of trade amount by what is possible
            trade_amount = min(self.remaining_energy, potential_trade)

            # If trade amount is 0 or less, cancel trade attempt
            if trade_amount <= 0:
                continue

            # Perform the trade
            self.remaining_energy -= trade_amount
            buyer.remaining_energy += trade_amount

            self.exported_energy += trade_amount
            buyer.energy_from_microgrid += trade_amount
            buyer.imported_energy += trade_amount

            # Stop if no more energy is available to trade
            if self.remaining_energy <= 0:
                break

    def _update_dataframe(self):
        """
        Updates the agent's statistics in the shared model data dictionary.

        This allows for tracking agent-level metrics over time for analysis or visualization.
        """
        # Retrieve the profile information
        self.profile = self.model.consumption_data[
            self.model.consumption_data["household"] == f"household_{self.unique_id}"
        ]["profiel"].values[0]

        # Gather the agents' data
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

        # Store the data in the model’s time-indexed data structure
        day_data = self.model.agent_data[self.model.day_str]
        day_data[self.model.hour][self.unique_id] = agent_data
