import os
import numpy as np
import matplotlib.pyplot as plt
from Network_Diff import *
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle

def load_trajectory(path, nrs):
    paths = []
    datas = []
    for nr in nrs:
        load_path = path + "push_trajectory" + str(nr) + ".npy"
        paths.append(load_path)
        data = np.load(load_path)
        datas.append(data) # return list of arrays

    datas = np.asarray(datas) # convert to array
    return datas

def calc_object_frame_x(x, y, theta):
    return (np.cos(theta) * x - np.sin(theta) * y)

def calc_object_frame_y(x, y, theta):
    return (np.sin(theta) * x + np.cos(theta) * y)

def get_training_data(data):
    # read all data:
    state_robot = data[:, 0:5]
    state_object = data[:, 5:11]
    acc_action = data[:, 11:13]
    time = data[:, 13]
    
    learning_x_all = []
    learning_x_true_all = []
    learning_x_true_next_all = []
    learning_y_all = []

    # for each timstep, transform data 
    for i in range(0, time.shape[0]-1):
        theta_o = state_object[i, 2]
        theta_r = state_robot[i, 2]

        vel_r_x = np.cos(theta_r) * state_robot[i, 3]
        vel_r_y = np.sin(theta_r) * state_robot[i, 3]
        vel_r_theta = state_robot[i, 4]
        vel_r = state_robot[i, 3]

        vel_o_x = state_object[i, 3]
        vel_o_y = state_object[i, 4]
        vel_o_theta = state_object[i, 5]

        pos_r_x = state_robot[i, 0]
        pos_r_y = state_robot[i, 1]
        pos_r_theta = state_robot[i, 2]

        pos_r_x_diff = state_robot[i, 0] - state_object[i, 0]
        pos_r_y_diff = state_robot[i, 1] - state_object[i, 1]
        pos_r_theta_diff = state_robot[i, 2] - state_object[i, 2]

        pos_o_x = state_object[i, 0]
        pos_o_y = state_object[i, 1]
        pos_o_theta = state_object[i, 2]

        acc_action_r = acc_action[i, 0]
        acc_action_theta = acc_action[i, 1]
        action_x = np.cos(theta_r) * acc_action[i, 0]
        action_y = np.sin(theta_r) * acc_action[i, 0]
        action_theta = acc_action[i, 1]

        learning_x  = np.array( [vel_r_x, vel_r_y, vel_r_theta, pos_r_x, pos_r_y, pos_r_theta,
         vel_o_x, vel_o_y, vel_o_theta,  pos_o_x, pos_o_y, pos_o_theta,
         action_x, action_y, action_theta]
          )

        d_px = state_object[i+1, 0] - state_object[i, 0]
        d_py = state_object[i+1, 1] - state_object[i, 1]
        d_theta = state_object[i+1, 2] - state_object[i, 2]
        d_vx = state_object[i+1, 3] - state_object[i, 3]
        d_vy = state_object[i+1, 4] - state_object[i, 4]
        d_omega = state_object[i+1, 5] - state_object[i, 5]

        # output data in the object frame
        learning_y = np.array( [ d_px, d_py, d_theta,
            d_vx, d_vy, d_omega
            ] )

        learning_x_true = np.array([pos_r_x, pos_r_y, pos_r_theta, vel_r, vel_r_theta, pos_o_x, pos_o_y, pos_o_theta, vel_o_x, vel_o_y, vel_o_theta, acc_action_r, acc_action_theta])

        vel_r_next = state_robot[i+1, 3]
        vel_r_theta_next = state_robot[i+1, 4]

        pos_r_x_next = state_robot[i+1, 0]
        pos_r_y_next = state_robot[i+1, 1]
        pos_r_theta_next = state_robot[i+1, 2]

        vel_o_x_next = state_object[i+1, 3]
        vel_o_y_next = state_object[i+1, 4]
        vel_o_theta_next = state_object[i+1, 5]

        pos_o_x_next = state_object[i+1, 0]
        pos_o_y_next = state_object[i+1, 1]
        pos_o_theta_next = state_object[i+1, 2]

        learning_x_true_next = np.array([pos_r_x_next, pos_r_y_next, pos_r_theta_next, vel_r_next, vel_r_theta_next, pos_o_x_next, pos_o_y_next, pos_o_theta_next, vel_o_x_next, vel_o_y_next, vel_o_theta_next])

        learning_x_all.append(learning_x)
        learning_y_all.append(learning_y)
        learning_x_true_all.append(learning_x_true)
        learning_x_true_next_all.append(learning_x_true_next)

    learning_x_all = np.asarray(learning_x_all)
    learning_y_all = np.asarray(learning_y_all)
    learning_x_true_all = np.asarray(learning_x_true_all)
    learning_x_true_next_all = np.asarray(learning_x_true_next_all)

    return learning_x_all, learning_y_all, learning_x_true_all, learning_x_true_next_all

def reshape_to_tensor(arr):
    arr = np.asarray(arr)
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    tens = torch.from_numpy(arr).float()
    return tens

def prepare_data(data):

    X = []
    y = []
    X_true = []
    X_true_next = []

    for i in range(data.shape[0]):
        calc_data = data[i, :, :]
        X_, y_, X_true_, X_true_next_ = get_training_data(calc_data)
        X.append(X_)
        y.append(y_)
        X_true.append(X_true_)
        X_true_next.append(X_true_next_)

    X = reshape_to_tensor(X)
    y = reshape_to_tensor(y)
    X_true = reshape_to_tensor(X_true)
    X_true_next = reshape_to_tensor(X_true_next)

    # return tensors of data in right shape
    return X, y, X_true, X_true_next


def main(total_data):
    # load training data (80 percent)
    max_ = int(total_data * 0.8) 
    lst_train = list(range(0,max_))
    load_path = "trajectories_train/" 
    data = load_trajectory(load_path, lst_train)

    X, y, X_true, X_true_next = prepare_data(data)

    # # scaling the data
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X.detach().numpy())
    # X = torch.from_numpy(X).float()
    # pickle.dump(scaler, open('/home/susan/Documents/isaac_nn_save/scaler.pkl','wb'))

    input_size = X.shape[1]
    output_size = y.shape[1]
    lr = 1e-4
    max_epochs = 100

    model = MLP(input_size, output_size, X, y, X_true, X_true_next)
    loss_function = nn.MSELoss() #nn.MSELoss() # nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-5) #weight_decay = 1e-5
    avg_loss = []

    for epoch in range(0, max_epochs): 
    
        batch_losses = []
        # Print epoch
        print(f'Starting epoch {epoch+1}')
                    
        # Set current loss value
        current_loss = 0.0
                    
        # Iterate over the DataLoader for training data
        dataiter = iter(model.loader)

        for batch in dataiter:
            inputs = batch[0]
            targets = batch[1]
            xtrue = batch[2]
            xtrue_next = batch[3]

            optimizer.zero_grad()
                                      
            # Perform forward pass
            outputs = model(inputs)
                                      
            # Compute loss
            loss = loss_function(outputs, targets)
                            
            # Perform backward pass
            loss.backward()
                                  
            # Perform optimization
            optimizer.step()
                                  
            # Print statistics
            current_loss = loss.item()
            batch_losses.append(current_loss)

        print("Avg epoch loss: ", sum(batch_losses)/len(batch_losses))
        avg_loss.append(sum(batch_losses)/len(batch_losses))

    plot_losses(avg_loss, max_epochs)
    
    torch.save(model.state_dict(), "saved_nn/torch_model_supervised_world.pt")
    print("Done training!")

def plot_losses(losses, max_epochs):
    epochs = list(range(1,max_epochs+1))
    # todo: validation losses
    plt.figure(742)   
    plt.plot(epochs, losses, color = "blue", label = "Train loss")
    plt.xlabel("Epochs]")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.savefig("losses_world.png")

def plot_trajectory(X_true, object_states, i):
    plt.figure(1*i)
    plt.plot(X_true[:, 5], X_true[:, 6], marker="o", color = "blue", label = "Actual object trajectory")
    plt.plot(X_true[:, 0], X_true[:, 1], marker="o", color = "red", label = "Actual robot trajectory")
    plt.plot(object_states[:, 0], object_states[:, 1], marker="x", color = "cyan", label = "Predicted object trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("plots_results_world/traj_trained_nn" + str(i) + ".png")

def plot_predictions(y_train, y_pred, i):
    plt.figure(1000000*i)
    plt.plot(y_train[:, 0], y_train[:, 1], "bo",  label = "Actual object diff")
    plt.plot(y_pred[:, 0], y_pred[:, 1], "ro",  label = "Predicted object diff")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig("plots_results_world/pred_x_y" + str(i) + ".png")

def update_X_input(new_object_state, X_true, n):
    # get true state for robot and action
    X_true_current = X_true[n, :].detach().numpy()

    state_robot = X_true_current[0:5]
    state_object = new_object_state
    acc_action = X_true_current[11:13]

    # calc all inputs
    theta_o = state_object[2]
    theta_r = state_robot[2]

    vel_r_x = np.cos(theta_r) * state_robot[3]
    vel_r_y = np.sin(theta_r) * state_robot[3]
    vel_r_theta = state_robot[4]

    vel_o_x = state_object[3]
    vel_o_y = state_object[4]
    vel_o_theta = state_object[5]

    pos_r_x = state_robot[0]
    pos_r_y = state_robot[1]
    pos_r_theta = state_robot[2]

    pos_o_x = state_object[0]
    pos_o_y = state_object[1]
    pos_o_theta = state_object[2]

    action_x = np.cos(theta_r) * acc_action[0]
    action_y = np.sin(theta_r) * acc_action[0]
    action_theta = acc_action[1]


    X_test_new  = np.array( [vel_r_x, vel_r_y, vel_r_theta, pos_r_x, pos_r_y, pos_r_theta,
         vel_o_x, vel_o_y, vel_o_theta,  pos_o_x, pos_o_y, pos_o_theta,
         action_x, action_y, action_theta]
          )

    # make tensor again
    X_test_new = torch.from_numpy(X_test_new).float()

    return X_test_new


def test(total_data):

    min_ = int(total_data * 0.8) 
    lst_test = list(range(min_, total_data))
    load_path = "trajectories_train/" 

    for i in lst_test:
        print("i: ", i)
        temp = [i]
        data = load_trajectory(load_path, temp)

        X_test, y_test, X_true, X_true_next = prepare_data(data)

        # scaler = pickle.load(open('/home/susan/Documents/isaac_nn_save/scaler.pkl','rb'))
        # X_test = scaler.transform(X_test.detach().numpy())
        # X_test = torch.from_numpy(X_test).float()

        X_true_plotting = X_true.detach().numpy()
        y_true_object = y_test.detach().numpy()

        input_size = X_test.shape[1]
        output_size = y_test.shape[1]
        model = MLP(input_size, output_size, X_test, y_test, X_true, X_true_next)
        model.load_state_dict(torch.load("saved_nn/torch_model_supervised_world.pt"))

        # Init the prediction with the TRUE first state 
        first_object_state = np.array([ X_true[0, 5], X_true[0, 6], X_true[0, 7], X_true[0, 8], X_true[0, 9], X_true[0, 10] ])
        object_state = first_object_state
        
        object_states = np.zeros((X_test.shape[0], first_object_state.shape[0]))
        object_states[0, :] = first_object_state

        # Init matrices to keep track of data
        y_pred_total = np.zeros((y_test.shape[0]-1, y_test.shape[1]))
        y_pred_total_world = np.zeros((y_test.shape[0]-1, y_test.shape[1]))

        y_test_total = np.zeros((y_test.shape[0]-1, y_test.shape[1]))
        X_pred_all = np.zeros((X_test.shape[0], X_test.shape[1]))

        # init first input for pred
        X_pred = X_test[0, :]

        print(X_pred, "x pred")
        mode = "previous"

        for n in range(0, y_test.shape[0]-1):
            # Predict the state diff in the object frame
            y_pred = model.forward(X_pred)
            y_pred = y_pred.detach().numpy()

            # Save data
            y_pred_total[n, :] = y_pred
            y_test_total[n, :] = y_test[n, :]
            print("pred: ", y_pred_total[n,:], "true: ", y_test[n, :])

            # Calc new object state 
            new_object_state = object_state + y_pred

            # Calc new X input for (n+1)
            X_pred_new = update_X_input(new_object_state, X_true, n+1)
            X_pred_all[n, :] = X_pred_new

            # Update
            object_state = new_object_state
            X_pred = X_pred_new

            object_states[n+1, :] = object_state

            mse = mean_squared_error(y_test[n, :], y_pred)
            print("mse: ", mse)

        plot_trajectory(X_true_plotting, object_states, temp[0])

        plot_predictions(y_test_total, y_pred_total, temp[0])



if __name__ == "__main__":
    total_data = 200
    # main(total_data)
    test(total_data)
