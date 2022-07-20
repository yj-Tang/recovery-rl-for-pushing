import numpy as np
from matplotlib import pyplot as plt
from collections import deque
import torch 

class MPPI_Push():

    def __init__(self, horizon, N):
        """
        initial_state: the starting position of the robot
        initial_action: an inital guess at the controls for the first horizon
        goal: target waypoint
        horizon: defines the time step, dt = 1/horizon
        lam: mppi parameter
        sig: sampling distribution varience
        N: number of rollouts
        """
        self.lam = 0.5
        self.sig_omega = 0.5
        self.sig_vel = 0.1

        self.N = N
        self.beta = 0.95
        self.horizon = horizon
        self.dt = 0.1

        self.input_feature_size = 12

        self.a = None #initial_action # 2xN
        self.a0 = None #initial_action[:,0] # action to append to a after robot has been issued a control
        self.cur_state = None #initial_state # 1x3 or just (3,)
        self.prev_state = None

        self.fin_path = None #[initial_state] # final path taken by the robot
        self.fin_control = [] # all of the control commands resulted in the final path
        self.fin_time = [0] # time stamp of the position/control
        self.goal = np.array([0.0, 0.0])

        self.max_input = 2.0

        self.t_cur = 0 # tracks the current time
        self.p = [self.prev_state]
        self.past_action = np.array([0.0, 0.0])

        self.model = None


    def init_matrices(self, x0):
        initial_state = x0[0:3] 
        initial_action = np.ones([2, self.horizon]) * np.array([[0.0, 0.0]]).T

        self.a = initial_action # 2xN
        self.a0 = initial_action[:,0] # action to append to a after robot has been issued a control
        self.prev_state = initial_state
        self.cur_state = initial_state # 1x3 or just (3,)

        self.fin_path = [initial_state] # final path taken by the robot

    def update_model(self, model):
        self.model = model

    def step(self, x, u, f):
        """
        RK4 integration step
        """

        k1 = f(x, u).T * self.dt
        k2 = f(x + 0.5*k1, u).T * self.dt
        k3 = f(x + 0.5*k2, u).T * self.dt
        k4 = f(x + k3, u).T * self.dt
        
        x_next = x + (k1 + 2*k2 + 2*k3 + k4)/6.0
        return x_next

    def update_robot_state_l(self, x, u):
        d_v_r = u[:, 0] 
        d_omega_r = u[:, 1]

        d_v_r = np.clip(d_v_r, -self.max_input, self.max_input)       
        d_omega_r = np.clip(d_omega_r, -self.max_input, self.max_input)

        px = x[:, 0]
        py = x[:, 1]
        theta = x[:, 2]
        v = x[:, 3]
        omega = x[:, 4]

        zeros = np.zeros_like(px)

        d_rob_state = np.array([v * np.cos(theta), v * np.sin(theta), omega, d_v_r, d_omega_r, zeros, zeros, zeros, zeros, zeros, zeros])
       
        return d_rob_state

    def update_robot_state(self, x, u):
        x = self.step(x, u, self.update_robot_state_l)
        return x

    def calc_object_frame_x(self, x, y, theta):
        return (np.cos(theta) * x - np.sin(theta) * y)

    def calc_object_frame_y(self, x, y, theta):
        return (np.sin(theta) * x + np.cos(theta) * y)

    def update_object_state_diff_object(self, x, u, x_prev): 
        d_v_r = u[:, 0] 
        d_omega_r = u[:, 1]

        px_r = x[:, 0]
        py_r = x[:, 1]
        theta_r = x[:, 2]
        v_r = x[:, 3]
        omega_r = x[:, 4]
      
        px_o = x[:, 5]
        py_o = x[:, 6]
        theta_o = x[:, 7]
        vx_o = x[:, 8]
        vy_o = x[:, 9]
        omega_o = x[:, 10]

        # calc states
        vel_r_x = np.cos(theta_r) * v_r
        vel_r_y = np.sin(theta_r) * v_r
        vel_r_theta = omega_r

        vel_r_x_oframe = self.calc_object_frame_x(vel_r_x, vel_r_y, -theta_o)
        vel_r_y_oframe = self.calc_object_frame_y(vel_r_x, vel_r_y, -theta_o)
        vel_r_theta_oframe = vel_r_theta

        vel_o_x = vx_o
        vel_o_y = vy_o
        vel_o_theta = omega_o

        vel_o_x_oframe = self.calc_object_frame_x(vel_o_x, vel_o_y, -theta_o)
        vel_o_y_oframe = self.calc_object_frame_y(vel_o_x, vel_o_y, -theta_o)
        vel_o_theta_oframe = vel_o_theta

        pos_r_x = px_r
        pos_r_y = py_r
        pos_r_theta = theta_r

        pos_o_x = px_o
        pos_o_y = py_o
        pos_o_theta = theta_o

        pos_r_x_diff = pos_r_x - pos_o_x
        pos_r_y_diff = pos_r_y - pos_o_y
        pos_r_theta_diff = pos_r_theta - pos_o_theta

        pos_r_x_oframe_diff = self.calc_object_frame_x(pos_r_x_diff, pos_r_y_diff, -theta_o)
        pos_r_y_oframe_diff = self.calc_object_frame_y(pos_r_x_diff, pos_r_y_diff, -theta_o)
        pos_r_theta_oframe_diff = pos_r_theta_diff

        acc_action_r = d_v_r
        acc_action_theta = d_omega_r
        action_x = np.cos(theta_r) * d_v_r
        action_y = np.sin(theta_r) * d_v_r
        action_theta = d_omega_r

        action_x_oframe = self.calc_object_frame_x(action_x, action_y, -theta_o)
        action_y_oframe = self.calc_object_frame_y(action_x, action_y, -theta_o)
        action_theta_oframe = action_theta

        # maybe in the learning do cos(theta) and sin(theta) 
        learning_x  = np.array( [vel_r_x_oframe, vel_r_y_oframe, vel_r_theta_oframe, pos_r_x_oframe_diff, pos_r_y_oframe_diff, pos_r_theta_oframe_diff,
         vel_o_x_oframe, vel_o_y_oframe, vel_o_theta_oframe, 
         action_x_oframe, action_y_oframe, action_theta_oframe]
          ).T

        # calc y_pred in object frame
        X_data = learning_x
        X_data = torch.from_numpy(X_data)
        y_pred = self.model.forward(X_data.float())
        y_pred = y_pred.detach().numpy()

        # then convert back to world frame
        d_px_o = self.calc_object_frame_x(y_pred[:, 0], y_pred[:, 1], theta_o)
        d_py_o = self.calc_object_frame_y(y_pred[:, 0], y_pred[:, 1], theta_o)
        d_theta_o = y_pred[:, 2]
        d_vx_o = self.calc_object_frame_x(y_pred[:, 3], y_pred[:, 4], theta_o)
        d_vy_o = self.calc_object_frame_y(y_pred[:, 3], y_pred[:, 4], theta_o)
        d_omega_o = y_pred[:, 5]

        zeros = np.zeros_like(px_o)
        d_obj_state = np.array([zeros, zeros, zeros, zeros, zeros, d_px_o, d_py_o, d_theta_o, d_vx_o, d_vy_o, d_omega_o]).T

        new_object_state = x + d_obj_state 

        return new_object_state


    def update_real_state(self, x, x_prev):
        self.cur_state = x
        self.prev_state = x_prev



    def l(self, x, a, eps):
        """
        cost function
        All inputs should correspond to the same time stamp
        x: the state of the robot
        a: the control taken
        eps: the sampled control pertubations
        """
        output = np.zeros([self.N, 1])

        a = a.reshape(1,2)
       
        for n in range(self.N):

            px_error = self.goal[0] - x[n,5]
            py_error = self.goal[1] - x[n,6]
            disToGoal = np.sqrt(px_error ** 2 + py_error ** 2)
            disToGoal = max(disToGoal, 0.2)
            cost_pos_navi = 1 * px_error ** 2 / disToGoal + 1 * py_error ** 2 / disToGoal

            px_error = x[n,0] - x[n,5]
            py_error = x[n,1] - x[n,6]
            disToGoal = np.sqrt(px_error ** 2 + py_error ** 2)
            disToGoal = max(disToGoal, 0.2)
            cost_pos_dist = 10 * px_error ** 2 / disToGoal + 10 * py_error ** 2 / disToGoal

            cost_husky_input = 10 * (a[0][0]) ** 2 + 10 * (a[0][1]) ** 2

            cost_husky_vel = 1 * (x[n,3]) ** 2 + 1 * (x[n,4]) ** 2

            output[n,:] = cost_husky_input + cost_pos_dist +  cost_husky_vel


        return output

   
    def m(self, x):
        """
        terminal cost function
        x: the state of the robot at the end of the horizon
        """
        output = np.zeros([self.N, 1])

        for n in range(self.N):

            px_error = self.goal[0] - x[n,5]
            py_error = self.goal[1] - x[n,6]
            disToGoal = np.sqrt(px_error ** 2 + py_error ** 2)
            disToGoal = max(disToGoal, 0.2)
            cost_pos_navi = 200 * px_error ** 2 / disToGoal + 200 * py_error ** 2 / disToGoal

            output[n,:] =  cost_pos_navi 
           
        return output


    def set_goal(self, x, y):
        """
        Function used to update the goal to drive towards
        """
        self.goal = np.array([x, y])


    def get_action(self):
        """
        Perform mppi algo, apply the control to the robot for 1 time step, and shift the control vecotr accordingly
        """
        J = [] # cost list
        eps = [] # samples list

        all_states = np.zeros((self.horizon, self.N, self.cur_state.shape[0]))
        temp_state = np.tile(self.cur_state, (self.N,1))
        prev_state = np.tile(self.prev_state, (self.N,1)) 

        eps_array = np.zeros((self.horizon, self.N, 2))
        vel_eps = np.random.normal(0, self.sig_vel, size=(self.horizon, self.N, 1))
        omega_eps = np.random.normal(0, self.sig_omega, size=(self.horizon, self.N, 1))
        eps_array[:, :, 0] = vel_eps.reshape(self.horizon, self.N)
        eps_array[:, :, 1] = omega_eps.reshape(self.horizon, self.N)

        eps_smoothed = np.zeros_like(eps_array)
        for i in range(self.horizon):
            if (i==0):
                eps_smoothed[i, :, :] = self.beta * (self.a[:, i] + eps_array[i, :, :]) + (1 - self.beta) * self.past_action
            else:
                eps_smoothed[i, :, :] = self.beta * (self.a[:, i] + eps_array[i, :, :]) + (1 - self.beta) * eps_smoothed[i-1, :, :]

        eps_smoothed = np.clip(eps_smoothed, -self.max_input, self.max_input)
        for t in range(self.horizon):

            J.append(self.l(temp_state, self.a[:,t], eps_smoothed[t, :, :]))

            self.p = prev_state

            new_temp_state = np.zeros_like(temp_state)

            new_husky_state = self.update_robot_state(temp_state,  eps_smoothed[t, :, :])
            new_object_state = self.update_object_state_diff_object(temp_state,  eps_smoothed[t, :, :], prev_state)
               
            new_temp_state[:, 0: 5] = new_husky_state[:, 0: 5]
            new_temp_state[:, 5:  ] = new_object_state[:, 5:  ]
     

            prev_state = temp_state
            temp_state = new_temp_state
            all_states[t, :, :] = temp_state

        self.mpc_path = all_states

        J.append(self.m(temp_state))


        J = np.flip(np.cumsum(np.flip(J, 0), axis=0), 0)

        for t in range(self.horizon):

            J[t] -= np.amin(J[t]) # log sum exp trick

            w = np.exp(-J[t]/self.lam) + 1e-8
            w /= np.sum(w)

            self.a[:,t] = self.a[:,t] + np.dot(w.T, (eps_smoothed[t, :, :]-self.a[:,t]))
                        
        # make sure action is between bounds
        action_lin_acc = np.clip(self.a[0,0], -self.max_input, self.max_input)
        action_omega_acc = np.clip(self.a[1,0], -self.max_input, self.max_input)
        action = np.array([action_lin_acc, action_omega_acc])

        self.past_action = action
        mpc_paths = self.mpc_path #(self.horizon, self.N, self.cur_state.shape[0])
        return action, mpc_paths


    def init_next_action(self):
        # advance control matrix for next step
        self.a = np.concatenate([self.a[:, 1:], np.array(self.a0).reshape(2,1)], axis=1)

def main():
    pass 

if __name__ == "__main__":
    main()
