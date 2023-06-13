import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files

data = pd.read_csv('benchmark_emissions.csv')
data = data.sort_values("CO2 Emissions (tonnes)").reset_index(drop="True")
df = pd.DataFrame(data)
X = list(df.iloc[:,0])
Y = list(df.iloc[:,1])
plt.barh(X, Y, color='grey')
plt.title("Comparing Emissions From Model Training and Other Actions")
plt.xlabel("Action")
plt.ylabel("C02 Emissions (tonnes)")
# Show the plot
plt.show()
plt.savefig('benchmark_emissions.pdf', format='pdf')
files.download('benchmark_emissions.pdf')

