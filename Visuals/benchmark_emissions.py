import matplotlib.pyplot as plt
import pandas as pd
import os 


data = pd.read_csv('../Datasets/benchmark_emissions.csv')
data = data.sort_values("CO2 Emissions (tonnes)").reset_index(drop="True")
# data.to_csv('../Datasets/modified_benchmark_emissions.csv')
# df.to_csv(filepath)  

def acc_visual(data):
    df = pd.DataFrame(data)
    X = list(df.iloc[:,0])
    Y = list(df.iloc[:,1])
    plt.barh(X, Y, color='grey')
    plt.title("Major sources of Carbon Emissions in tonnes")
    plt.xlabel("CO2 Emissions (in tonnes)")
    plt.ylabel("Sources")
    # plt.legend(frameon=False)
    # Show the plot
    plt.savefig('benchmark_emissions.pdf',dpi='figure',bbox_inches="tight")


acc_visual(data)