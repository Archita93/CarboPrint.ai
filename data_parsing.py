import pandas as pd
from pandasql import sqldf

gpus_dataset = pd.read_csv("Datasets/gpu_modified.csv")
gamma_data = pd.read_csv("Datasets/gamma_modified.csv")
pue_dataset = pd.read_csv("Datasets/pue_modified.csv")

def data_parse(user_input,dataframe):
    query = ""
    try: 
        if dataframe == "gpu_dataset":
            query = "SELECT G.tdp_watts FROM gpus_dataset AS G WHERE G.name LIKE '{}'".format(user_input)
        elif dataframe == "gamma_data":
            query = "SELECT G.Gamma FROM gamma_data AS G WHERE G.Regions LIKE '{}'".format(user_input)
        elif dataframe == "pue_dataset":
            query = "SELECT G.PUE FROM pue_dataset AS G WHERE G.Providers LIKE '{}'".format(user_input)
        result = sqldf(query, env=None)
        resultant = float(result.iloc[0][0])
        return resultant
    except:
        print("The particular input is not available in the dataset")
        return None


# result = data_parse("AGX Xavier","gpu_dataset")
# print(type(result))

# result2 = data_parse("Africa","gamma_data")
# print(result2)

# result2 = data_parse("Google Cloud","pue_dataset")
# print(result2)



def calculate_carbon_emissions(num_of_chips,region, providers,grid_config, hours_trained, energy, num_processors, avg_power_per_processor,hardware_type):
    total_energy = 0
    
    if energy:
        total_energy = energy
        
    else:
        val1 = data_parse(hardware_type,"gpu_dataset")
        if val1:
            total_energy = (hours_trained * val1 * num_of_chips)

        else:
            if num_processors and avg_power_per_processor:
                total_energy = hours_trained * num_processors * avg_power_per_processor

        total_energy = float(total_energy/1000) # kWh 
    val2 = data_parse(region,"gamma_data")
    if val2:
        gamma = val2
    else:
        gamma = 378.625  # g/Kwh
    gamma = float(gamma/1000) # kg/Kwh
    
    val3 = data_parse(providers,"pue_dataset")
    if val3:   # no units
        PUE = val3
    else:
        PUE = 1
    
    carbon_emissions = gamma * PUE * (total_energy)
    if grid_config and grid_config > 0:
        return float((carbon_emissions*grid_config)/1000)
    else:
        return float(carbon_emissions/1000)

# result = calculate_carbon_emissions(10000,"United States of America","Microsoft Azure",1,355.2,None,None,None,"Tesla V100-SXM2-32GB")
# print(result)







