def estimate_carbon_emissions(PUE, E_CPU, E_GPU, E_RAM, gamma):
    total_energy = PUE * (E_CPU + E_GPU + E_RAM)
    carbon_footprint = gamma * total_energy
    return carbon_footprint

# Prompt user for input or calculate total energy if individual energy values are not available
use_individual_energy = input("Do you have individual energy consumption values? (Y/N): ")

if use_individual_energy.lower() == "y":
    E_CPU = float(input("Enter the energy consumed by CPUs (kWh): "))
    E_GPU = float(input("Enter the energy consumed by GPUs (kWh): "))
    E_RAM = float(input("Enter the energy consumed by RAM (kWh): "))
else:
    hours_to_train = float(input("Enter the number of hours to train: "))
    num_processors = int(input("Enter the number of processors: "))
    average_power_per_processor = float(input("Enter the average power per processor (W): "))

    total_energy = (hours_to_train * num_processors * average_power_per_processor) / 1000  # convert to kWh
    E_CPU = total_energy / 3  # Assuming equal energy distribution
    E_GPU = total_energy / 3
    E_RAM = total_energy / 3

PUE = float(input("Enter the Power Usage Effectiveness (PUE): "))
gamma = float(input("Enter the gamma coefficient (kg/kWh): "))

carbon_emissions = estimate_carbon_emissions(PUE, E_CPU, E_GPU, E_RAM, gamma)
print(f"The estimated carbon emissions are {carbon_emissions} kg.")
