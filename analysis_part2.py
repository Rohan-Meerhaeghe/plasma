import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import scipy.constants as cst


class Results:
    def __init__(self) -> None:
        self.I1 = np.zeros((2, 3), dtype="float")
        self.dI1dV = np.zeros((2, 3), dtype="float")
        self.I2 = np.zeros((2, 3), dtype="float")
        self.dI2dV = np.zeros((2, 3), dtype="float")
        self.dIdV = np.zeros((2, 3), dtype="float")
        self.Te = np.zeros((2,3), dtype="float")


results = Results()


class Data:
    def __init__(self, data, pressure, current):
        self.data = data
        self.pressure = pressure
        if current.endswith(".txt"):
            self.current = current[:-4]
        else:
            self.current = current


def floating_potential(voltage, current):
    abs_current = np.abs(current)
    zero_index = np.argmin(abs_current)
    second_zero_index = zero_index + np.argmax(np.transpose(abs_current[zero_index-1:zero_index+1])) - 1
    floating_pot = (abs_current[zero_index]*voltage[second_zero_index]+abs_current[second_zero_index]*voltage[zero_index])/(abs_current[zero_index]+abs_current[second_zero_index])
    return voltage - np.full_like(voltage, floating_pot)


def read_txt(name):
    with open(name, "r") as file:
        txtReader = pandas.read_csv(file, sep="\t", decimal=".", skiprows=1)
        raw_data = np.transpose(np.array(txtReader))
        raw_data[0] = floating_potential(raw_data[0], raw_data[1])
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


def fit_asymptotes(
    xdata,
    ydata,
    pressure,
    current,
    first_a_guess=1,
    first_b_guess=-2,
    second_a_guess=1,
    second_b_guess=2,
):
    zero_index = np.argmin(np.abs(xdata))

    stop_index_0 = round(zero_index / 2)
    popt_0, pcov_0 = curve_fit(
        asymptote,
        xdata[:stop_index_0],
        ydata[:stop_index_0],
        [first_a_guess, first_b_guess],
    )

    start_index_1 = round((len(xdata) - zero_index) / 2 + zero_index)
    popt_1, pcov_1 = curve_fit(
        asymptote,
        xdata[start_index_1:],
        ydata[start_index_1:],
        [second_a_guess, second_b_guess],
    )

    start_index_2 = zero_index - 5
    stop_index_2 = zero_index + 5
    popt_2, pcov_2 = curve_fit(
        asymptote,
        xdata[start_index_2:stop_index_2],
        ydata[start_index_2:stop_index_2],
        [1, 0],
    )
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(xdata, ydata, "b-", label="Data")
    plt.xlabel("Probe voltage [V]", loc="right")
    plt.ylabel("Probe current [mA]", loc="top")
    plt.plot(
        xdata[:zero_index],
        asymptote(xdata[:zero_index], *popt_0),
        "r--",
        label=r"asymptote $I_{2+}$",
    )
    plt.plot(
        xdata[zero_index:],
        asymptote(xdata[zero_index:], *popt_1),
        "g--",
        label=r"asymptote $I_{1+}$",
    )
    plt.plot(
        xdata[start_index_2 - 10 : stop_index_2 + 10],
        asymptote(xdata[start_index_2 - 10 : stop_index_2 + 10], *popt_2),
        "m--",
        label="tangent at origin",
    )

    plt.legend()

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Eliminate upper and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.savefig("langmuir_probe_p_" + pressure + "_i_" + current + ".png")
    plt.close()
    """
    print("\tPressure: ", pressure, " mbar, current: ", current," mA")
    print("I1+: ",-popt_1[1])
    print("dI1+/dV: ", popt_1[0])
    print("I2+: ", popt_0[1])
    print("dI2+/dV: ", popt_0[0])
    print("dI/dV: ", popt_2[0])
    """

    if pressure == "0.28mbar":
        index_0 = 0
    else:
        index_0 = 1
    if current == "4mA":
        index_1 = 0
    elif current == "8mA":
        index_1 = 1
    else:
        index_1 = 2

    results.I1[index_0][index_1] = -popt_1[1]
    results.dI1dV[index_0][index_1] = popt_1[0]
    results.I2[index_0][index_1] = popt_0[1]
    results.dI2dV[index_0][index_1] = popt_0[0]
    results.dIdV[index_0][index_1] = popt_2[0]
    results.Te[index_0][index_1] = cst.elementary_charge/cst.Boltzmann*(2*np.abs(popt_1[1])*popt_0[1]/(-popt_1[1]+popt_0[1])*1/(2*popt_2[0]-0.5*(popt_1[0]+popt_0[0])))

for data_set in data:
    fit_asymptotes(
        data_set.data[0], data_set.data[1], data_set.pressure, data_set.current
    )


def string_formatter(result_data):
    return (
        "&  4 mA&  8 mA& 15 mA\\\\ \hline \n\t0.08 mbar&  "
        + np.array2string(result_data[0], separator="& ",precision=3)[1:-1]
        + "\\\\ \hline \n\t0.28 mbar&  "
        + np.array2string(result_data[1], separator="& ",precision=3)[1:-1]
        + "\\\\ \hline\n"
    )


print("\tI1+:\n", string_formatter(results.I1))
print("\tdI1+/dV: \n", string_formatter(results.dI1dV))
print("\tI2+: \n", string_formatter(results.I2))
print("\tdI2+/dV: \n", string_formatter(results.dI2dV))
print("\tdI/dV: \n", string_formatter(results.dIdV))
print("\tTe: \n", string_formatter(results.Te))