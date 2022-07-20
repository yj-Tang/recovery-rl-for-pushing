import os
import numpy as np
import matplotlib.pyplot as plt

def load_trajectory(path):
    data = np.load(path)
    data = data[data.shape[0] - 20:, :] # skip first row of zeros
    return data

def save_updated_trajectory(data, save_path):
    np.save(save_path, data)

def plot_trajectory(data, i):
    # read all data:
    state_robot = data[:, 0:5]
    state_object = data[:, 5:11]
    acc_action = data[:, 11: 13]
    time = data[:, 13]

    # plotting robot + object + contact point trajectories
    plt.figure(0)
    plt.plot(state_object[:, 0], state_object[:, 1], marker="o", color = "blue", label = "Object trajectory") 
    plt.plot(state_robot[:, 0], state_robot[:, 1], marker="o", color = "magenta", label = "Robot trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("plots/trajectories_" + str(i) + ".png")

def inspect_vel_acc_data(data, i):
    # read all data:
    state_robot = data[:, 0:5]
    state_object = data[:, 5:11]
    acc_action = data[:, 11: 13]
    time = data[:, 13]

    # plotting robot velocity
    plt.figure(1)
    plt.plot(time, state_robot[:, 3], marker="o", color = "red", label = "X velocity")
    # plt.plot(time, state_robot[:, 4], marker="o", color = "blue", label = "Y velocity") 
    plt.plot(time, state_robot[:, 4], marker="o", color = "magenta", label = "Theta velocity")
    plt.xlabel("t [s]")
    plt.ylabel("vel [m/s] [rad/s]")
    plt.legend()
    plt.savefig("plots/robot_velocity_" + str(i) + ".png")

    # plotting object velocity
    plt.figure(2)
    plt.plot(time, state_object[:, 3], marker="o", color = "red", label = "X velocity")
    plt.plot(time, state_object[:, 4], marker="o", color = "blue", label = "Y velocity") 
    plt.plot(time, state_object[:, 5], marker="o", color = "magenta", label = "Theta velocity")
    plt.xlabel("t [s]")
    plt.ylabel("vel [m/s] [rad/s]")
    plt.legend()
    plt.savefig("plots/object_velocity_" + str(i) + ".png")

    # plotting acceleration action
    plt.figure(6)
    plt.plot(time, acc_action[:, 0], marker="o", color = "red", label = "Acc action t")
    plt.plot(time, acc_action[:, 1], marker="o", color = "blue", label = "Acc action omega") 
    plt.xlabel("t [s]")
    plt.ylabel("acc [m2/s] [rad2/s]")
    plt.legend()
    plt.savefig("plots/acceleration_action_" + str(i) + ".png")

def main():

    max_traj = 200
    plotting = False
    for i in range(0, max_traj):
        name = "push_trajectory" + str(i) + ".npy"
        load_path = "./trajectories_collect/" + str(name) 
        save_path = "./trajectories_train/" + str(name) 
        data = load_trajectory(load_path)

        # plot 
        if plotting == True:
            plot_trajectory(data, i)
            inspect_vel_acc_data(data, i)

        # savind cropped
        save_updated_trajectory(data, save_path)

if __name__ == "__main__":
    main()
