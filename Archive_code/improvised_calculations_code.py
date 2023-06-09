import pandas as pd

# Read the GPU dataset from CSV
gpus_dataset = pd.read_csv("gpus.csv")
gpus_dataset = gpus_dataset[["name", "type", "tdp_watts"]]

# Read the gamma dataset from CSV
gamma_dataset = pd.read_csv("gamma.csv")
gamma_data = gamma_dataset.drop('Country code', axis=1)
idx = gamma_data.groupby('Country or region')['Year'].idxmax()
gamma_data = gamma_data.loc[idx]
gamma_data = gamma_data.rename(columns={'Country or region': 'Regions', 'Emissions intensity (gCO2 per kWh)': 'Gamma'})

# Read the PUE dataset from CSV
pue_dataset = pd.read_csv("pue.csv")
pue_dataset = pue_dataset.rename(columns={'Cloud Provider': 'Providers'})

def data_parse(user_input, dataframe):
    if dataframe == "gpus_dataset":
        result = gpus_dataset[gpus_dataset["name"].str.contains(user_input, na=False, case=False)]["tdp_watts"].values
    elif dataframe == "gamma_data":
        result = gamma_data[gamma_data["Regions"].str.contains(user_input, na=False, case=False)]["Gamma"].values
    elif dataframe == "pue_dataset":
        result = pue_dataset[pue_dataset["Providers"].str.contains(user_input, na=False, case=False)]["Power Usage Effectiveness (PUE)"].values
    return float(result[0]) if len(result) > 0 and not pd.isna(result[0]) else None




def calculate_carbon_emissions(PUE, E_CPU, E_GPU, E_RAM, gamma, hours_trained, num_processors, avg_power_per_processor):
    """
    Calculate carbon emissions using the formula: CF = γ · PUE · (E_CPU + E_GPU + E_RAM)
    Args:
        PUE: Power Usage Effectiveness
        E_CPU: Energy consumption of CPU (optional)
        E_GPU: Energy consumption of GPU (optional)
        E_RAM: Energy consumption of RAM (optional)
        gamma: Gamma coefficient
        hours_trained: Hours of training
        num_processors: Number of processors
        avg_power_per_processor: Average power per processor per hour
    Returns:
        The calculated carbon emissions in kg.
    """
    if E_CPU is None and E_GPU is None and E_RAM is None:
        total_energy = hours_trained * num_processors * avg_power_per_processor
        E_CPU = total_energy / 3
        E_GPU = total_energy / 3
        E_RAM = total_energy / 3

    carbon_emissions = gamma * PUE * (E_CPU + E_GPU + E_RAM)
    return carbon_emissions

# Read the GPU dataset from CSV
gpus_dataset = pd.read_csv("gpus.csv")
gpus_dataset = gpus_dataset[["name", "tdp_watts"]]

# Read the gamma dataset from CSV
gamma_dataset = pd.read_csv("gamma.csv")

# Read the PUE dataset from CSV
pue_dataset = pd.read_csv("pue.csv")

# Retrieve user inputs (selected values from front end)
user_gpu_name = "Selected GPU Name"
user_region = "Selected Region"
user_provider = "Selected Provider"
hours_trained = 100
num_processors = 8
avg_power_per_processor = 150

# Retrieve energy consumption values from GPU dataset
gpu_row = gpus_dataset[gpus_dataset["name"].str.contains(user_gpu_name, case=False)]
E_CPU = float(gpu_row["tdp_watts"]) if len(gpu_row) > 0 else None

# Retrieve gamma coefficient value from gamma dataset
gamma_row = gamma_dataset[gamma_dataset["Regions"].str.contains(user_region, case=False)]
gamma = float(gamma_row["Gamma"]) if len(gamma_row) > 0 else None

# Retrieve PUE value from PUE dataset
pue_row = pue_dataset[pue_dataset["Providers"].str.contains(user_provider, case=False)]
PUE = float(pue_row["Power Usage Effectiveness (PUE)"]) if len(pue_row) > 0 else None

# Check if any values could not be retrieved
missing_values = []
if E_CPU is None:
    missing_values.append("E_CPU")
if gamma is None:
    missing_values.append("gamma")
if PUE is None:
    missing_values.append("PUE")

if len(missing_values) > 0:
    print(f"Error: Some values could not be retrieved: {', '.join(missing_values)}")
    exit()

# Calculate carbon emissions
carbon_emissions = calculate_carbon_emissions(PUE, E_CPU, None, None, gamma, hours_trained, num_processors, avg_power_per_processor)

print(f"The estimated carbon emissions are {carbon_emissions} kg.")
