# DONE odstranit veličiny s vysokou korelací
# PARTIALLY DONE upravit stavový vzorec, získat optimální řízení, 
# DONE kontrola, jak počítat dojezd
# DONE koeficient penalizace, 
# DONE debugging optimálního řízení, 
# TODO model prvního řádu
# DONE test norm dat - někde rozdíl, někde ne

# DONE odstranit voltage
# DONE smes modelu
# DONE dojezd - problém - kde vzít kapacitu
# DONE zkoumat setpointy

# TODO jen výpočet dojezdu


from model import BayesReg

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

global keys
keys = [
        'Throttle [%]', # někde lze vynechat, sníží se však nepřesnost odhadu
       #  'Elevation [m]', 
       #  'Heating Power CAN [kW]',
       # 'AirCon Power [kW]',
       ]

def get_curr_model(df, showPlot=False):
    print("Training current model...")

    X = np.array(
        df[
            keys
        ]
    )

    nd = df.shape[0]

    reg_ride = BayesReg(len(keys), 2, intercept=True)
    reg_stop = BayesReg(len(keys), 2, intercept=True)
    delayed = np.zeros(reg_ride.delayed_val.shape[0])
    reg_stop.delayed_val = delayed
    reg_ride.delayed_val = delayed

    split = 0.75
    train_size = round(nd * split)

    curr = df["Battery Current [A]"]
    currp = np.zeros(train_size)
    currv = np.zeros(nd - train_size)


    for t in range(1, train_size):
        if df["Velocity [km/h]"][t] > 3:
            currp[t] = reg_ride.predict_update(X[t, :], curr[t])
        else:
            currp[t] = reg_stop.predict_update(X[t, :], curr[t])

    for t in range(train_size, nd):
        if df["Velocity [km/h]"][t] > 3:
            currv[t - train_size] = reg_ride.predict(X[t, :], True)
        else:
            currv[t - train_size] = reg_stop.predict(X[t, :], True)

    reg_ride.deleteDelayed()
    reg_stop.deleteDelayed()


    print(f"RMSE: {root_mean_squared_error(curr[train_size:],currv):.3f}")
    print(f"R2: {r2_score(curr[train_size:],currv):.6f}")
    print(f"Coef: {reg_ride.Eth}")

    # Plot
    if showPlot:
        plt.plot(df["Time [s]"], curr, "b:", label="current")
        plt.plot(df["Time [s]"][:train_size], currp, "g:", label="training")
        plt.plot(df["Time [s]"][train_size:], currv, "r:", label="validation")
        plt.xlabel("Time Step [s]")
        plt.ylabel("Current [A]")
        plt.legend()
        plt.show()

    return reg_ride, reg_stop

if __name__ == "__main__":
    # kapacita baterie vzána ze specifikace daného elektroauta, 
    # podle aktuálních výpočtů nesouhlasí s kapacitou elektroauta v datech
    battery_capacity = 60 * 3600 # [Q] = As = C

    datafile = f"dataKaggle/Trip{input('Enter trip code: ')}.csv"

    df = pd.read_csv(datafile, encoding="iso8859-1", delimiter=";")
    df.dropna(axis=1,inplace=True)


    reg_ride, reg_stop = get_curr_model(df, showPlot=True)
    nd = df.shape[0]

    ## Optimal control for current


    rng = np.random.default_rng()
    for t in range(1, round(rng.random() * nd)):
        if df["Velocity [km/h]"][t] > 2:
            reg_ride.predict(X[t, :], True)
        else:
            reg_stop.predict(X[t, :], True)

    setpoints = np.concatenate((np.linspace(10,-100,1001), np.linspace(-100, 10 ,1001)))

    throttle_opt, current_opt = reg_ride.optimal_control(
        setpoints.size, om=0.1, la=1.0, setpoint=setpoints, 
        upper=100, lower=0
    )

    current = df["Battery Current [A]"][: round(nd / 2)]
    throttle = df["Throttle [%]"][: round(nd / 2)]
    velocity = df["Velocity [km/h]"][: round(nd / 2)]

    start_plot_from = 4700

    _, ax = plt.subplots(figsize=(16, 12))
    plt.plot(
        np.arange(1, current.shape[0] + 1 - start_plot_from),
        current[start_plot_from:],
        "b:",
        label="current",
    )
    plt.plot(
        np.arange(1, current.shape[0] + 1 - start_plot_from),
        throttle[start_plot_from:],
        "g:",
        label="throttle",
    )
    plt.plot(
        np.arange(1, current.shape[0] + 1 - start_plot_from),
        velocity[start_plot_from:],
        "y:",
        label="velocity",
    )
    plt.plot(
        np.arange(
            current.shape[0] + 1 - start_plot_from,
            current.shape[0] + 1 - start_plot_from + current_opt.shape[0],
        ),
        current_opt,
        "r:",
        label="current_opt",
    )
    plt.plot(
        np.arange(
            current.shape[0] + 1 - start_plot_from,
            current.shape[0] + 1 - start_plot_from + current_opt.shape[0],
        ),
        throttle_opt,
        "m:",
        label="throttle_opt",
    )
    plt.xlabel("Time Step [s]")
    plt.ylabel("Speed [m/s]")
    plt.legend()
    plt.show()


