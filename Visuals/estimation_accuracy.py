import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ed1 = pd.read_csv("../Datasets/estimate_1.csv")
ed2 = pd.read_csv("../Datasets/estimate_2.csv")

ed1 = ed1.sort_values("CO2(tonnes)").reset_index(drop="True")
ed2 = ed2.sort_values("CO2(tonnes)").reset_index(drop="True")
# ed = ed.dropna(how='all').transpose().reset_index()
# estimate_data = ed.rename({'index':'Model',0:'Datacenter',1:'Gamma',2:'PUE',3:'Processor',4:'TDP_WATTS',5:'chip_num',6:'training_time',7:'Energy(KWh)',8:'CO2(tonnes)'}, axis='columns')
# estimate_data.drop(estimate_data.index[0], inplace=True)
# estimate_data = estimate_data.astype({'Gamma':'float','PUE':'float','TDP_WATTS':'float','chip_num':'float','training_time':'float'})
# estimate_data.to_csv("estimation_data.csv",index= False)


def carbon_emissions(gamma,pue,tdp_watts,chip_num,training_time):
    energy = tdp_watts*chip_num*training_time
    energy = float(energy/1000)
    carbon_estimates = energy*gamma*pue
    return float(carbon_estimates/1000000)

def ready_data(data):
    data['CO2_Estimates'] = data.apply(lambda x: carbon_emissions((x['Gamma']), (x['PUE']),(x['TDP_WATTS']),(x['chip_num']),(x['training_time'])), axis=1)
    estimate_plot = data[["Model","CO2(tonnes)","CO2_Estimates"]]
    estimate_plot = estimate_plot.astype({'CO2_Estimates':'float','CO2(tonnes)':'float'})
    return estimate_plot

# import matplotlib.pyplot as plt
# import numpy as np

def visual(data,num):
    # Plot using Matplotlib
    fig, ax = plt.subplots()
    estimate_plot = ready_data(data)
    # Set the x positions of the bars
    x = np.arange(len(estimate_plot['Model']))

    # Split the data based on the threshold
    width = 0.35

    # Plot the actual quantities
    ax.bar(x - width/2, estimate_plot['CO2(tonnes)'], width, label='Actual')

    # Plot the estimated quantities
    ax.bar(x + width/2, estimate_plot['CO2_Estimates'], width, label='Estimated')


    # Set the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(estimate_plot['Model'],rotation=45,ha='right')

    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('CO2 Emissions \n(in tonnes)', rotation=0)
    ax.set_title('Comparison between Actual and Estimated Quantities')


    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig('accuracy_plot'+str(num)+'.pdf',dpi='figure')  # Save the graph as PNG image
   

visual(ed1,1)
visual(ed2,2)


