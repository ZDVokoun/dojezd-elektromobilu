import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from current_model import get_curr_model
from velocity_model import get_vel_model
import cmd

battery_capacity = 60 # [Q] = As = C

def plot(current_opt, throttle_opt, velocity_opt):
    # plt.plot(
    #     np.arange(1, current.shape[0] + 1),
    #     current[:],
    #     "b:",
    #     label="current",
    # )
    # plt.plot(
    #     np.arange(1, current.shape[0] + 1),
    #     throttle[:],
    #     "g:",
    #     label="throttle",
    # )
    # plt.plot(
    #     np.arange(1, current.shape[0] + 1),
    #     velocity[:],
    #     "y:",
    #     label="velocity",
    # )
    plt.plot(
        np.arange(
            0,
            current_opt.shape[0],
        ),
        current_opt,
        "r:",
        label="Optimal current",
    )
    plt.plot(
        np.arange(
            0,
            current_opt.shape[0],
        ),
        throttle_opt,
        "m:",
        label="Optimal throttle",
    )
    plt.plot(
        np.arange(
            0,
            current_opt.shape[0],
        ),
        velocity_opt,
        "c:",
        label="Optimal velocity",
    )
    # plt.xlabel("Time Step [s]")
    # plt.ylabel("")
    plt.legend()
    plt.show()

def velocity_optimal(reg_ride, vel_reg, a, b, c):
    throttle_opt_conc = []
    current_opt_coc = []
    velocity_opt_conc = []

    for v in range(a, b, c):
        n = 1500
        setpoints = np.ones(n) * v

        throttle_opt, velocity_opt = vel_reg.optimal_control(
            setpoints.size, om=0.1, la=1.0, setpoint=setpoints, 
            upper=100, lower=0
        )

        current_opt = np.zeros(n)
        for i in range(n):
            current_opt[i] = reg_ride.predict([throttle_opt[i]], True)
        throttle_opt_conc.append(throttle_opt)
        throttle_opt_conc.append(np.zeros(100))
        current_opt_coc.append(current_opt)
        current_opt_coc.append(np.zeros(100))
        velocity_opt_conc.append(velocity_opt)
        velocity_opt_conc.append(np.zeros(100))
        vel_reg.deleteDelayed()
        print(f"Range with {current_opt[-1]:.2f} A and {velocity_opt[-1]:.2f} km/h: {(- np.average(velocity_opt) / np.average(current_opt) * battery_capacity):.2f} km")

    current_opt = np.concatenate(current_opt_coc[:-1])
    velocity_opt = np.concatenate(velocity_opt_conc[:-1])
    throttle_opt = np.concatenate(throttle_opt_conc[:-1])
    plot(current_opt, throttle_opt, velocity_opt)


def current_optimal(reg_ride, vel_reg, a, b, c):
    throttle_opt_conc = []
    current_opt_coc = []
    velocity_opt_conc = []

    for I in range(a, b, c):
        n = 1500
        setpoints = np.ones(n) * I

        throttle_opt, current_opt = reg_ride.optimal_control(
            setpoints.size, om=0.1, la=1.0, setpoint=setpoints, 
            upper=100, lower=0
        )

        velocity_opt = np.zeros(n)
        for i in range(n):
            velocity_opt[i] = vel_reg.predict([throttle_opt[i]], True)
        throttle_opt_conc.append(throttle_opt)
        throttle_opt_conc.append(np.zeros(100))
        current_opt_coc.append(current_opt)
        current_opt_coc.append(np.zeros(100))
        velocity_opt_conc.append(velocity_opt)
        velocity_opt_conc.append(np.zeros(100))
        vel_reg.deleteDelayed()
        print(f"Range with {I} A and {velocity_opt[-1]:.2f} km/h: {(- np.average(velocity_opt) / np.average(current_opt) * battery_capacity):.2f} km")

    current_opt = np.concatenate(current_opt_coc[:-1])
    velocity_opt = np.concatenate(velocity_opt_conc[:-1])
    throttle_opt = np.concatenate(throttle_opt_conc[:-1])
    plot(current_opt, throttle_opt, velocity_opt)


def throttle_optimal(reg_ride, vel_reg, a, b, c):
    throttle_opt_conc = []
    current_opt_coc = []
    velocity_opt_conc = []

    for thr in range(a, b, c):
        n = 1500
        throttle_opt = np.ones(n) * thr
        current_opt = np.zeros(n)
        velocity_opt = np.zeros(n)
        for i in range(n):
            velocity_opt[i] = vel_reg.predict([throttle_opt[i]], True)
            current_opt[i] = reg_ride.predict([throttle_opt[i]], True)
        throttle_opt_conc.append(throttle_opt)
        throttle_opt_conc.append(np.zeros(100))
        current_opt_coc.append(current_opt)
        current_opt_coc.append(np.zeros(100))
        velocity_opt_conc.append(velocity_opt)
        velocity_opt_conc.append(np.zeros(100))
        vel_reg.deleteDelayed()
        print(f"Range with {current_opt[-1]:.2f} A and {velocity_opt[-1]:.2f} km/h: {(- np.average(velocity_opt) / np.average(current_opt) * battery_capacity):.2f} km")

    current_opt = np.concatenate(current_opt_coc[:-1])
    velocity_opt = np.concatenate(velocity_opt_conc[:-1])
    throttle_opt = np.concatenate(throttle_opt_conc[:-1])
    plot(current_opt, throttle_opt, velocity_opt)

global reg_ride, reg_stop, vel_reg

def parse(arg):
    'Convert a series of zero or more numbers to an argument tuple'
    return tuple(map(int, arg.split()))

class CLI(cmd.Cmd):
    prompt = ">>> "
    intro = "Enter 'help' to see the usage of commands. To exit enter 'exit'"
    reg_ride = None
    reg_stop = None
    vel_reg = None

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except AttributeError:
            print("Ilegal action. Have you loaded a dataset?")
        except (ValueError, IndexError):
            print("Ilegal action. Have you entered correct command or parameters?")
        except FileNotFoundError:
            print("File not found. Have you entered correct file name or saved dataset files into correct directory?")
        return False # don't stop
    def do_exit(self, line):
        """exit: Exits the program"""
        return True
    def do_load(self, line):
        """load code: Loads dataset with 'Trip{code}.csv' (this has to be run first)"""
        datafile = f"dataKaggle/Trip{line}.csv"

        df = pd.read_csv(datafile, encoding="iso8859-1", delimiter=";")
        df.dropna(axis=1,inplace=True)
        self.reg_ride, self.reg_stop = get_curr_model(df, showPlot=False)
        self.vel_reg = get_vel_model(df)
    def do_velocity_optimal(self, line):
        """velocity_optimal s: Sets the velocity setpoint to value s, runs optimal control algorithm and prints vehicle range """
        inp = parse(line)
        velocity_optimal(self.reg_ride, self.vel_reg, inp[0], inp[0] + 1, 1)
    def do_throttle_optimal(self, line):
        """throttle_optimal s: Sets the throttle setpoint to value s, runs optimal control algorithm and prints vehicle range"""
        inp = parse(line)
        throttle_optimal(self.reg_ride, self.vel_reg, inp[0], inp[0] + 1, 1)
    def do_current_optimal(self, line):
        """current_optimal s: Sets the current setpoint to value s, runs optimal control algorithm and prints vehicle range"""
        inp = parse(line)
        current_optimal(self.reg_ride, self.vel_reg, inp[0], inp[0] + 1, 1)
    def do_velocity_optimal_range(self, line):
        """velocity_optimal_range a b c: Runs velocity_optimal for values in 'range(a, b, c)'"""
        inp = parse(line)
        velocity_optimal(self.reg_ride, self.vel_reg, inp[0], inp[1], inp[2])
    def do_throttle_optimal_range(self, line):
        """throttle_optimal_range a b c: Runs throttle_optimal for values in 'range(a, b, c)'"""
        inp = parse(line)
        throttle_optimal(self.reg_ride, self.vel_reg, inp[0], inp[1], inp[2])
    def do_current_optimal_range(self, line):
        """current_optimal_range a b c: Runs current_optimal for values in 'range(a, b, c)'"""
        inp = parse(line)
        current_optimal(self.reg_ride, self.vel_reg, inp[0], inp[1], inp[2])


if __name__ == "__main__":
    reg_ride = None
    reg_stop = None
    vel_reg = None
    CLI().cmdloop()
