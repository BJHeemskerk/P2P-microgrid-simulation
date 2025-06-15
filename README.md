# **P2P Energy Trading in Microgrids using Agent-Based Modelling**

This repository contains the code, documents and research for a research project that was part of the course DataLab V in 2025. In this project, we explore the possibilities in which the dynamics of peer-to-peer energy trading within a microgrid can be simulated using Agent-Based Modelling.


## Table of contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Project structure](#project-structure)
- [Usage](#usage)
- [The Team](#the-team)


## Introduction

The energy transition requires a fundamental revision of the current, largely centralized energy system. Traditionally, electricity was generated in large-scale power plants and distributed through a hierarchical network. However, there is now a rapid increase in decentralized renewable energy sources. This development, combined with the electrification of sectors such as mobility and industry, puts increasing pressure on the existing grid and poses complex challenges for grid operators.

A promising innovation in this evolving energy landscape is Peer-to-Peer (P2P) energy trading. This model allows consumers and producers to trade electricity directly through digital platforms without the intervention of traditional energy suppliers. Research indicates that P2P trading is attractive to consumers due to potential cost savings, increased energy autonomy, and more sustainable energy use. However, large-scale implementation is hindered by limitations in the current network infrastructure.

Microgrids — small-scale, local energy networks — represent a relevant technological development to facilitate P2P trading. According to IRENA, microgrids not only enable more efficient exchange of locally generated energy but also contribute to lower energy costs and a more robust, resilient energy system.

Since setting up a physical microgrid is costly and time-consuming, simulating microgrids with ABM offers an accessible and cost-effective way to gain insights into the effects of P2P trading. By modeling households as individual agents with distinct behaviors and energy profiles, it becomes possible to investigate under which conditions a microgrid operates efficiently, stably, and cost-effectively. This enables drawing conclusions about optimal scenarios without the need for direct investment in physical infrastructure.


## Dependencies

This project was developed using Jupyter notebooks and requires Python 3.8+.

### Required Python Libraries

- `mesa` — Agent-Based Modelling framework  
- `numpy` — Numerical operations  
- `pandas` — Data manipulation  
- `matplotlib` — Plotting and visualization  
- `seaborn` — Statistical data visualization  
- `python-dateutil` — Advanced date utilities  

### Installation via pip

You can install all dependencies with the following command:

```bash
pip install -r requirements.txt
```

To run the simulation, open the notebook in a Jupyter environment after installing the dependencies or run gather_results.py after installing the dependencies.


## Project structure
```plaintext
data/                      # Input data files
  ├── solar_strength.csv
  └── solar_2years.csv

Notebooks/                 # Jupyter notebooks for data analysis
  ├── Dataset_Power.ipynb
  └── energie_consumptie_N.ipynb

results/                   # Simulation results organized in folders
  └── [Test-run 1]/   # Each folder contains:
      ├── Agent_data.csv
      ├── Agent_net_energy.png
      ├── Daily_Price_comparison.png
      ├── Impact_battery.png
      ├── Import_export.png
      ├── simulation_data.csv
      └── simulation_report.txt

scripts/                   # Python modules for simulation and analysis
  ├── agent.py
  ├── gather_results.py
  ├── model.py
  ├── results.py
  └── utils.py

.gitignore                 # Git ignore file
Mesa_Testing.ipynb         # Main tutorial/testing notebook with working code
README.md                  # Project documentation (this file)
requirements.txt           # Python dependencies
```


## Usage

1. Clone the repository:
```bash
git clone https://github.com/BJHeemskerk/DataLabV_ABM.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open and run `Mesa_Testing.ipynb` to explore the simulation and analysis.
5. Optionally, run `scripts/gather_results.py` to aggregate data from simulation outputs.


## The Team

- Jasper Duncker
- Tim Oosterling
- Busse Heemskerk


