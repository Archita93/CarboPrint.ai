!pip install ipynb
import pandas as pd
ed = pd.read_csv("estimation_accuracy.csv")
ed = ed.dropna(how='all').transpose().reset_index()
estimate_data = ed.rename({'index':'Model',0:'Datacenter',1:'Gamma',2:'PUE',3:'Processor',4:'TDP_WATTS',5:'chip_num',6:'training_time',7:'Energy(KWh)',8:'CO2(tonnes)'}, axis='columns')
estimate_data.drop(estimate_data.index[0], inplace=True)
estimate_data.head(10)

estimate_data = estimate_data.astype({'Gamma':'float','PUE':'float','TDP_WATTS':'float','chip_num':'float','training_time':'float'})
def carbon_emissions(gamma,pue,tdp_watts,chip_num,training_time):
    energy = tdp_watts*chip_num*training_time
    energy = float(energy/1000)
    carbon_estimates = energy*gamma*pue
    return float(carbon_estimates/1000000)

estimate_data['CO2_Estimates'] = estimate_data.apply(lambda x: carbon_emissions((x['Gamma']), (x['PUE']),(x['TDP_WATTS']),(x['chip_num']),(x['training_time'])), axis=1)
estimate_data.head(10)

estimate_plot = estimate_data[["Model","CO2(tonnes)","CO2_Estimates"]]
estimate_plot = estimate_plot.astype({'CO2_Estimates':'float','CO2(tonnes)':'float'})

estimate_plot.head(10)

import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

def visual():
    # Set the threshold for splitting the graph
    threshold = 0.5  # Adjust this value as per your requirement

    # Split the data based on the threshold
    low_values = estimate_plot[estimate_plot['CO2(tonnes)'] <= threshold]
    high_values = estimate_plot[estimate_plot['CO2(tonnes)'] > threshold]

    # Set the width of the bars
    width = 0.25  # Adjust this value to decrease or increase the width

    # Set the x positions of the bars
    x_low = np.arange(len(low_values['Model']))
    x_high = np.arange(len(high_values['Model']))

    # Plot the low values
    plt.figure(figsize=(8, 6))
    plt.bar(x_low - width/2, low_values['CO2(tonnes)'], width, label='Actual', color='blue')
    plt.bar(x_low + width/2, low_values['CO2_Estimates'], width, label='Estimated', color='orange')
    plt.xticks(x_low, low_values['Model'], rotation=45, ha='right')
    plt.xlabel('Models')
    plt.ylabel('Carbon Dioxide \n Emissions \n (in tonnes)', rotation=0, ha='right')
    plt.title('Comparison between Actual and Estimated Quantities (Low Values)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('low_values_graph.pdf', format='pdf')  # Save the graph as PNG image
    plt.show()
    files.download('low_values_graph.pdf')

    # Plot the high values with reduced size and width
    plt.figure(figsize=(5, 4))
    plt.bar(x_high - width/2, high_values['CO2(tonnes)'], width, label='Actual', color='blue')
    plt.bar(x_high + width/2, high_values['CO2_Estimates'], width, label='Estimated', color='orange')
    plt.xticks(x_high, high_values['Model'], rotation=45, ha='right')
    plt.xlabel('Models')
    plt.ylabel('Carbon Dioxide \n Emissions \n (in tonnes)', rotation=0, ha='right')
    plt.title('Comparison between Actual and Estimated Quantities (High Values)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('high_values_graph.pdf', format='pdf')
    plt.show()
    files.download('high_values_graph.pdf')

visual()


from sklearn.preprocessing import StandardScaler

# Create a MinMaxScaler object
scaler = StandardScaler()

# Scale the 'Actual' column
estimate_plot['CO2(tonnes)'] = scaler.fit_transform(estimate_plot['CO2(tonnes)'].values.reshape(-1, 1))

# Scale the 'Estimated' column
estimate_plot['CO2_Estimates'] = scaler.fit_transform(estimate_plot['CO2_Estimates'].values.reshape(-1, 1))
