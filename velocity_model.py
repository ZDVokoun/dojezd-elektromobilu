from model import BayesReg

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

global keys
keys = [
        'Throttle [%]', 
        # 'Ambient Temperature [°C]',
        # 'AirCon Power [kW]',
        # 'Battery Temperature [°C]',
        # 'Elevation [m]',
        # "Velocity [km/h]",
]

def get_vel_model(df, showPlot=False):
    print("Training velocity model...")
    nd = df.shape[0]
    acc = np.zeros(nd)
    delayed = np.zeros(nd)
    v = np.array(df["Velocity [km/h]"])
    delayed[:-1] = v[1:]
    acc[:-1] = np.diff(v)
    df["Acc"] = acc
    df["Delayed"] = delayed



    X = np.array(
        df[
            keys
        ]
    )

    reg_ride = BayesReg(len(keys), 2, intercept=False)

    split = 0.75
    train_size = round(nd * split)

    vel = df["Delayed"]
    velp = np.zeros(train_size)

    for t in range(1, train_size):
        velp[t] = reg_ride.predict_update(X[t, :], vel[t])

    velv = np.zeros(nd - train_size)

    for t in range(train_size, nd - 1):
        velv[t - train_size] = reg_ride.predict(X[t, :], True)
        if velv[t - train_size] < 0:
            velv[t - train_size] = 0
        # X[t+1,1] = velv[t - train_size]


    print(f"RMSE: {root_mean_squared_error(vel[train_size:],velv):.3f}")
    print(f"R2: {r2_score(vel[train_size:],velv):.6f}")
    print(f"Coef: {reg_ride.Eth}")

    # Plot
    if showPlot:
        plt.plot(df["Time [s]"], vel, "b:", label="current")
        plt.plot(df["Time [s]"][:train_size], velp, "g:", label="training")
        plt.plot(df["Time [s]"][train_size:], velv, "r:", label="validation")
        plt.xlabel("Time Step [s]")
        plt.ylabel("Current [A]")
        plt.legend()
        plt.show()

    return reg_ride

if __name__ == "__main__":
    datafile = f"dataKaggle/Trip{input('Enter trip code: ')}.csv"

    df = pd.read_csv(datafile, encoding="iso8859-1", delimiter=";")
    df.dropna(axis=1,inplace=True)

    get_vel_model(df, True)
