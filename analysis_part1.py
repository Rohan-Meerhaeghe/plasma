import pandas
import numpy as np
import matplotlib.pyplot as plt

def read_csv(name):
    with open(name, "r") as file:
        csvReader = pandas.read_csv(file, delimiter=";", decimal=",", skiprows=1)
        raw_data = np.transpose(np.array(csvReader))
        return raw_data

plasma_028mbar = read_csv("plasma 0.28mbar.csv")
plasma_0053mbar = read_csv("plasma 0.053mbar.csv")

fig, ax1 = plt.subplots() 
ax1.set_xlabel('Current [mA]') 
ax1.set_ylabel('System voltage [V]', color = 'red') 
ax1.plot(plasma_0053mbar[0], plasma_0053mbar[1], color = 'red') 
ax1.tick_params(axis ='y', labelcolor = 'red') 
  
# Adding Twin Axes
ax2 = ax1.twinx()
ax2.set_ylabel('Voltage between probes [V]', color = 'blue') 
ax2.plot(plasma_0053mbar[0], plasma_0053mbar[2], color = 'blue') 
ax2.tick_params(axis ='y', labelcolor = 'blue')
plt.savefig("plasma 0.0053mbar.png")
plt.close()


fig, ax1 = plt.subplots() 
ax1.set_xlabel('Current [mA]') 
ax1.set_ylabel('System voltage [V]', color = 'red') 
ax1.plot(plasma_028mbar[0], plasma_028mbar[1], color = 'red') 
ax1.tick_params(axis ='y', labelcolor = 'red') 
  
# Adding Twin Axes
ax2 = ax1.twinx()
ax2.set_ylabel('Voltage between probes [V]', color = 'blue') 
ax2.plot(plasma_028mbar[0], plasma_028mbar[2], color = 'blue') 
ax2.tick_params(axis ='y', labelcolor = 'blue')
plt.savefig("plasma 0.028mbar.png")
plt.close()
