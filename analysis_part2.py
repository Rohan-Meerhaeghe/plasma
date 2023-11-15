import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class Data():
    def __init__(self, data, pressure, current):
        self.data = data
        self.pressure = pressure
        if (current.endswith(".txt")):
            self.current = current[:-4]
        else:
            self.current = current

def floating_potential(voltage, current):
    zero_index = np.argmin(np.abs(current))
    return voltage-np.full_like(voltage,voltage[zero_index])

def read_txt(name):
    with open(name, "r") as file:
        txtReader = pandas.read_csv(file, sep="\t", decimal=".", skiprows=1)
        raw_data = np.transpose(np.array(txtReader))
        raw_data[0] = floating_potential(raw_data[0],raw_data[1])
        return Data(raw_data, name.split("_")[3], name.split("_")[5])


directory = "D:/Users/EorlTheYoung/Documents/Ugent/2023-2024/plasma/"

data = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename.endswith(".txt"):
        data.append(read_txt(filename.split("/")[-1]))

def asymptote(t, a, b):
    return a * t + b

def fit_asymptotes(xdata,ydata,pressure,current,first_a_guess=1,first_b_guess=-2,second_a_guess=1,second_b_guess=2):
    zero_index = np.argmin(np.abs(xdata))
    
    stop_index_0 = round(zero_index/2)
    popt_0, pcov_0 = curve_fit(asymptote, xdata[:stop_index_0], ydata[:stop_index_0], [first_a_guess,first_b_guess])

    start_index_1 = round((len(xdata)-zero_index)/2+zero_index)
    popt_1, pcov_1 = curve_fit(asymptote, xdata[start_index_1:], ydata[start_index_1:], [second_a_guess,second_b_guess])

    start_index_2 = zero_index - 10
    stop_index_2 = zero_index + 10
    popt_2, pcov_2 = curve_fit(asymptote, xdata[start_index_2:stop_index_2],ydata[start_index_2:stop_index_2],[1,0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(xdata, ydata, 'b-',label='Data')
    plt.xlabel('Probe voltage [V]',loc='right')
    plt.ylabel('Probe current [mA]',loc='top')
    plt.plot(xdata[:zero_index], asymptote(xdata[:zero_index], *popt_0), 'r--', label='asymptote I2+')
    plt.plot(xdata[zero_index:], asymptote(xdata[zero_index:], *popt_1), 'g--', label='asymptote I1+')
    plt.plot(xdata[start_index_2:stop_index_2], asymptote(xdata[start_index_2:stop_index_2], *popt_2), 'm--', label='tangent at origin')

    plt.legend()

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig("langmuir_probe_p_" + pressure + "_i_" + current + ".png")
    plt.close()
    print("\tPressure: ", pressure, " mbar, current: ", current," mA")
    print("I1+: ",popt_1[1])
    print("dI1+/dV: ", popt_1[0])
    print("I2+: ", popt_0[1])
    print("dI2+/dV: ", popt_0[0])
    print("dI/dV: ", popt_2[0])

for data_set in data:
    fit_asymptotes(data_set.data[0], data_set.data[1], data_set.pressure, data_set.current)