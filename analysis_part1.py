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

V_plasma_028mbar = plasma_028mbar[1] - plasma_028mbar[0] * 100
# current in mA and resistor in kOhm, so the milli and kilo cancel.
V_plasma_0053mbar = plasma_0053mbar[1] - plasma_0053mbar[0] * 100


plt.plot(plasma_0053mbar[0], V_plasma_0053mbar, "b-", label="P = 0.053 mbar")
plt.xlabel("Current [mA]")
plt.ylabel("Voltage accros plasma tube [V]")
plt.plot(plasma_028mbar[0], V_plasma_028mbar, "r-", label="P = 0.28 mbar")
plt.legend()
plt.savefig("plasma_voltage_tube.png")
plt.show()

plt.plot(plasma_028mbar[0], plasma_028mbar[2], "b-", label="P = 0.28 mbar")
plt.plot(plasma_0053mbar[0], plasma_0053mbar[2], "r-", label = "P = 0.053 mbar")
plt.xlabel("Current [mA]")
plt.ylabel("Voltage between probes [V]")
plt.legend()
plt.savefig("plasma_voltage_probes.png")
plt.show()