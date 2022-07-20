import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, RegularPolygon
from scipy import integrate
from scipy.interpolate import interp1d
import math

from utils.integrator import my_RK4
from general_problem_setup import Problem, Index
import utils.utils as utils
from dynamics.cylinder_rob_push_dynamics import cylinder_pushing_dynamics

class cylinder_robot_pushing_recBox(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human'], "render_fps": 30}

    def __init__(self):
        super(cylinder_robot_pushing_recBox, self).__init__()

        self.pr = Problem()
        self.index = Index()

        self.time = 0.0
        self.diata_t = 0.1

        # system setup
        self.rob_size = self.pr.rob_size    # m (radius of the cylinder)
        self.obj_size = self.pr.obj_size    # m
        self.obj_mass = self.pr.obj_mass    # kg
        self.obj_I_or = self.pr.obj_I_or    # moment of inertia (rotation center is 
                                            # the center of the object) (for rectangle)
        self.mu_g = self.pr.mu_g            # friction coefficient of the ground-object contact surface
        self.grav_acc = self.pr.grav_acc    # gravity acceleration

        # state initialization
        obj_pos_w = np.random.uniform([-2.4, -2.4], [2.4, 2.4])       # x, y
        # if obj_pos_w[1] > 0: 
        #     theta_o = np.random.uniform(-np.pi/2.0, 0.0)
        #     # theta_o = 0.0
        # else:
        #     theta_o = np.random.uniform(0.0, np.pi/2.0)

        opt_angle = math.atan2(-obj_pos_w[1], -obj_pos_w[0])
        theta_o = np.random.uniform( opt_angle - np.deg2rad(30), opt_angle + np.deg2rad(30))
        robot_pos_o = np.array([- self.rob_size - 0.5*self.obj_size[0], np.random.uniform(-0.5*self.obj_size[1], 0.5*self.obj_size[1])])
        robot_pos_w = obj_pos_w[0:2].T + np.dot(utils.rot_M(theta_o), robot_pos_o.T)      # position of the robot in the world frame
        theta_r = theta_o #math.atan2(-robot_pos_w[1], -robot_pos_w[0])


        v_r = 0.1
        vx_r = np.cos(theta_r) * v_r
        vy_r = np.sin(theta_r) * v_r
        d_theta_r = 0.0
        vx_o = 0.001
        vy_o = 0.00
        d_theta_o = (-np.cos(theta_o) * (vx_r - vx_o) - np.sin(theta_o) * (vy_r - vy_o) )/ \
                    (np.cos(theta_o)*(robot_pos_w[1] - obj_pos_w[1]) - np.sin(theta_o)*(robot_pos_w[0] - obj_pos_w[0]) ) 

        self.obj_state = np.array([obj_pos_w[0], obj_pos_w[1], theta_o, vx_o, vy_o, d_theta_o])
        self.rob_state = np.array([robot_pos_w[0], robot_pos_w[1], theta_r, v_r, d_theta_r])

        self.prev_rob_state = self.rob_state
        self.prev_obj_state = self.obj_state

        # initialize object history path
        self.obj_his_path = np.array(self.obj_state[0:3]).reshape((-1, 1))

        # visualization initialization, range of the x-y axises
        self.ws_x = self.pr.ws_x
        self.ws_y = self.pr.ws_y

        # prepare for visualization
        self.prepare_visualization()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space_low = np.array([self.pr.robot_minAcc, self.pr.robot_minOmegaAcc])    # robot and object state
        self.action_space_high = np.array([self.pr.robot_maxAcc, self.pr.robot_maxOmegaAcc])   
        self.action_space = spaces.Box(self.action_space_low, self.action_space_high, dtype=np.float32)
        
        self.observation_space_low = np.array([self.pr.ws_x[0], self.pr.ws_y[0], -1.5 * np.pi, self.pr.robot_minVel, -self.pr.robot_maxOmega, \
                                                self.pr.ws_x[0], self.pr.ws_y[0], -1.5 * np.pi, self.pr.robot_minVel, -self.pr.robot_maxOmega])     # robot and object state
        self.observation_space_high =  np.array([self.pr.ws_x[1], self.pr.ws_y[1], 1.5 * np.pi, self.pr.robot_maxVel, self.pr.robot_maxOmega, \
                                                self.pr.ws_x[1], self.pr.ws_y[1], 1.5 * np.pi, self.pr.robot_maxVel, self.pr.robot_maxOmega])    # robot and object state
        self.observation_space = spaces.Box(low=self.observation_space_low, high=self.observation_space_high, dtype=np.float32)

        # initialize the system state
        self.state = np.concatenate([self.rob_state, self.obj_state], 0)
        self.prev_state = self.state
        self.done = False
        self.success = False
        self.constraint = False

        # init action
        self.action = np.array([0.0, 0.0])
        self.input_size = 14
        self.full_trajectory = np.zeros(self.input_size).reshape(1, -1)
        self.mpc_paths = None

    def step(self, action):
        # system state update
        # d_state = cylinder_pushing_dynamics(self.state, action, params=NAN)

        self.prev_rob_state = self.rob_state
        self.prev_obj_state = self.obj_state
        self.prev_state = self.state

        self.state = my_RK4(np.array(self.state),\
                            np.array(action),\
                            cylinder_pushing_dynamics, self.diata_t, [])

        self.rob_state = self.state[self.index.rob_state]    # 0, 1, 2, 3, 4
        self.obj_state = self.state[self.index.obj_state]   # 5, 6, 7, 8, 9, 10

        self.obj_his_path = np.concatenate((self.obj_his_path, np.array(self.obj_state[0:3]).reshape((-1, 1))),
                                                 axis=1)
        self.action = action
        self.state = np.concatenate([self.rob_state, self.obj_state], 0)
        self.save_trajectory()
        self.time += self.diata_t

        # distance from the object to the goal
        distObjGoal = self.obj_state[0]**2 + self.obj_state[1]**2
        reward = -distObjGoal
        done = False 
        obj_pos_w_x = self.obj_state[0]
        obj_pos_w_y = self.obj_state[1]
        theta_o = self.obj_state[2]
        robot_pos_w_x = self.rob_state[0]
        robot_pos_w_y = self.rob_state[1]
        self.pos_c = [-0.25,- np.sin(theta_o) * (robot_pos_w_x -obj_pos_w_x) + np.cos(theta_o) * (robot_pos_w_y-obj_pos_w_y)]     # position of the contact point expressed in the object frame
        
        theta_r = self.rob_state[2]
        phi = theta_o - theta_r
        d = self.pos_c[1]
        if d > self.pr.d_max or d < -self.pr.d_max or phi > self.pr.phi_max or phi < -self.pr.phi_max:
            done = True
            self.constraint = True

        info = {
            "constraint": self.constraint,
            "reward": reward,
            "state": self.prev_state,
            "next_state": self.state,
            "action": action,
            "success": self.success,
        }

        return np.array(self.state, dtype=np.float32), reward, done, info


    def reset(self):
        # Reset the state of the environment to an initial state
        obj_pos_w = np.random.uniform([-2.4, -2.4], [2.4, 2.4])       # x, y
        distRobotGoal = np.sqrt(obj_pos_w[0]**2 + obj_pos_w[1]**2)
        while distRobotGoal < 1.5:
            obj_pos_w = np.random.uniform([-2.4, -2.4], [2.4, 2.4]) 

            distRobotGoal = np.sqrt(obj_pos_w[0]**2 + obj_pos_w[1]**2)
        # if obj_pos_w[1] > 0: 
        #     theta_o = np.random.uniform(-np.pi/2.0, 0.0)
        #     # theta_o = 0.0
        # else:
        #     theta_o = np.random.uniform(0.0, np.pi/2.0)

        opt_angle = math.atan2(-obj_pos_w[1], -obj_pos_w[0])
        theta_o = np.random.uniform( opt_angle - np.deg2rad(30), opt_angle + np.deg2rad(30))
        robot_pos_o = np.array([- self.rob_size - 0.5*self.obj_size[0], np.random.uniform(-0.5*self.obj_size[1], 0.5*self.obj_size[1])])
        robot_pos_w = obj_pos_w[0:2].T + np.dot(utils.rot_M(theta_o), robot_pos_o.T)      # position of the robot in the world frame
        theta_r = theta_o #math.atan2(-robot_pos_w[1], -robot_pos_w[0])

        v_r = 0.1
        vx_r = np.cos(theta_r) * v_r
        vy_r = np.sin(theta_r) * v_r
        d_theta_r = 0.0
        vx_o = 0.001
        vy_o = 0.00
        d_theta_o = (-np.cos(theta_o) * (vx_r - vx_o) - np.sin(theta_o) * (vy_r - vy_o) )/ \
                    (np.cos(theta_o)*(robot_pos_w[1] - obj_pos_w[1]) - np.sin(theta_o)*(robot_pos_w[0] - obj_pos_w[0]) ) 


        self.obj_state = np.array([obj_pos_w[0], obj_pos_w[1], theta_o, vx_o, vy_o, d_theta_o])
        self.rob_state = np.array([robot_pos_w[0], robot_pos_w[1], theta_r, v_r, d_theta_r])
        # test
        # self.obj_state = np.array([-1.55427822960784,1.9390335428066474,-0.3041396536467986,0.001,0.0,0.6884649339485412])
        # self.rob_state = np.array([-1.9742544985088042,1.9366912923778516,-0.7757938238982371,0.1,0.0])

        # initialize object history path
        self.obj_his_path = np.array(self.obj_state[0:3]).reshape((-1, 1))

        # initialize the system state
        self.state = np.concatenate([self.rob_state, self.obj_state], 0)
        self.done = False

        # reset robot traj
        self.full_trajectory = np.zeros(self.input_size).reshape(1, -1)
        self.time = 0

        self.done = False
        self.success = False
        self.constraint = False

        self.prev_state = self.state

        return self.state
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # Update robot current pos 
        self.fig_robot_pos.set_center([self.rob_state[0], self.rob_state[1]])

        # Update object current pos 
        pos_corner_o = np.array(self.obj_size).T / -2.0   # position of the corner in the object frame
        pos_corner_w = self.obj_state[0:2].T + np.dot(utils.rot_M(self.obj_state[2]), pos_corner_o)      # position of the corner in the world frame
        self.fig_object_pos.set_xy([pos_corner_w[0], pos_corner_w[1]])     # the left bottom coner position 
        self.fig_object_pos.angle = np.rad2deg(self.obj_state[2])
        self.fig_object_pos.set_width(self.obj_size[0])
        self.fig_object_pos.set_height(self.obj_size[1])

        # Update object history path
        self.fig_object_path[0].set_data(np.concatenate((self.obj_his_path[0, :], self.obj_his_path[1, :])).reshape((2, -1)))

        # update contact surface
        point1_o = np.array([-self.obj_size[0]/2.0,  self.obj_size[1]])
        point2_o = np.array([-self.obj_size[0]/2.0, -self.obj_size[1]])
        point1_w = np.dot(utils.rot_M(self.obj_state[2]), point1_o.T) + np.array([self.obj_state[0], self.obj_state[1]])
        point2_w = np.dot(utils.rot_M(self.obj_state[2]), point2_o.T) + np.array([self.obj_state[0], self.obj_state[1]])
        self.fig_contact_surface[0].set_data([point1_w[0], point2_w[0]], [point1_w[1], point2_w[1]])

        # Update canvas
        self.fig_main.canvas.draw()
        self.fig_main.canvas.flush_events()
        
    def prepare_visualization(self):
        # ================== prepare for visualization ====================
        # Prepare a figure for visualization 
        plt.ion()
        self.fig_main, self.ax_main = plt.subplots()
        self.ax_main.grid(visible=True, ls='-.')
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlim(self.ws_x)
        self.ax_main.set_ylim(self.ws_y)
        self.ax_main.set_xlabel('x [m]')
        self.ax_main.set_ylabel('y [m]')

        # plot objects
        # robot current pos
        robot_pos_cir = mpatches.Circle(np.array([self.rob_state[0],self.rob_state[1]]), self.rob_size, fc=(0, 0.4, 1, 0.8), ec=(0, 0, 1, 0.8), alpha = 0.4)
        self.fig_robot_pos = self.ax_main.add_artist(robot_pos_cir)

        # object current pos
        pos_corner_o = np.array(self.obj_size).T / -2.0   # position of the corner in the object frame
        object_pos_rec = mpatches.Rectangle([pos_corner_o[0], pos_corner_o[1]], self.obj_size[0], self.obj_size[1], 0.0, fc=(0, 1, 1, 0.1), ec=(0, 0, 1, 0.8))
        self.fig_object_pos = self.ax_main.add_artist(object_pos_rec)

        # object history path 
        self.fig_object_path = self.ax_main.plot(0.0, 0.0, c='g', ls='-', lw=2.0)

        plt.draw() 

        # contact surface
        point1_o = np.array([-self.obj_size[0]/2.0,  self.obj_size[1]*2])
        point2_o = np.array([-self.obj_size[0]/2.0, -self.obj_size[1]*2])
        point1_w = np.dot(utils.rot_M(self.obj_state[2]), point1_o.T) + np.array([self.obj_state[0], self.obj_state[1]])
        point2_w = np.dot(utils.rot_M(self.obj_state[2]), point2_o.T) + np.array([self.obj_state[0], self.obj_state[1]])
        self.fig_contact_surface = self.ax_main.plot([point1_w[0], point2_w[0]], [point1_w[1], point2_w[1]], linestyle = 'dotted')
        # ========================================================================


    def init_path_candidates(self, N):
        self.obj_mpc_planned_path = ['path_i']*N
        for i in range(N):
            self.obj_mpc_planned_path[i] = self.ax_main.plot(0.0, 0.0, c="red", ls='-', lw=2.0)

    def show_paths_candidates(self, mpc_path):
        all_x_coord = []
        all_y_coord = []

        for i in range(0, mpc_path.shape[1]):
            x_coord = mpc_path[:, i, 0]  # a list of x in one episode
            y_coord = mpc_path[:, i, 1]  # a list of y in one episode
            all_x_coord.append(x_coord)
            all_y_coord.append(y_coord)

        # # first remove all paths
        # for paths in self.obj_mpc_planned_path:
        #     paths.remove()
        # now draw new planned paths
        for j in range(len(self.obj_mpc_planned_path)):
            # self.obj_mpc_planned_path.append(self.ax_main.plot(all_x_coord[j], all_y_coord[j], c="red", ls='-', lw=2.0))
            self.obj_mpc_planned_path[j][0].set_data(np.concatenate((all_x_coord[j], all_y_coord[j])).reshape((2, -1)))

        # Update canvas
        self.fig_main.canvas.draw()
        self.fig_main.canvas.flush_events()

    def init_path_candidates_both(self, N):
        self.obj_mpc_planned_path = ['path_i']*N
        self.obj_mpc_planned_path_object = ['path_i']*N
        for i in range(N):
            self.obj_mpc_planned_path[i] = self.ax_main.plot(0.0, 0.0, c="red", ls='-', lw=2.0)
            self.obj_mpc_planned_path_object[i] = self.ax_main.plot(0.0, 0.0, c="magenta", ls='-', lw=2.0)

    def show_paths_candidates_both(self, mpc_path):
        all_x_coord = []
        all_y_coord = []
        all_x_coord_object = []
        all_y_coord_object = []

        for i in range(0, mpc_path.shape[1]):
            x_coord = mpc_path[:, i, 0]  # a list of x in one episode
            y_coord = mpc_path[:, i, 1]  # a list of y in one episode
            all_x_coord.append(x_coord)
            all_y_coord.append(y_coord)

            x_coord_object = mpc_path[:, i, 5]  # a list of x in one episode
            y_coord_object = mpc_path[:, i, 6]  # a list of y in one episode
            all_x_coord_object.append(x_coord_object)
            all_y_coord_object.append(y_coord_object)

        # # first remove all paths
        # for paths in self.obj_mpc_planned_path:
        #     paths.remove()
        # now draw new planned paths
        for j in range(len(self.obj_mpc_planned_path)):
            # self.obj_mpc_planned_path.append(self.ax_main.plot(all_x_coord[j], all_y_coord[j], c="red", ls='-', lw=2.0))
            self.obj_mpc_planned_path[j][0].set_data(np.concatenate((all_x_coord[j], all_y_coord[j])).reshape((2, -1)))
            self.obj_mpc_planned_path_object[j][0].set_data(np.concatenate((all_x_coord_object[j], all_y_coord_object[j])).reshape((2, -1)))

        # Update canvas
        self.fig_main.canvas.draw()
        self.fig_main.canvas.flush_events()

    def save_trajectory(self):
        # action [acc_r, acc_omega]
        # robot state: self.rob_state [x, y, theta, vr, omega]
        # object state: self.obj_state [x, y, theta, vx, vy, omega]

        full_state = np.append(np.append(np.append(self.prev_rob_state, self.prev_obj_state), self.action), self.time)
        full_state = full_state.reshape(1, -1)
        self.full_trajectory = np.concatenate((self.full_trajectory, full_state), axis=0)

    def save_trajectory_data(self, i_episode):
        np.save('/home/susan/Documents/_isaac/sim_yu/trajectories/trajectories_collect/push_trajectory' + str(i_episode) + '.npy', self.full_trajectory)

    def set_mpc_paths(self, mpc_paths):
        self.mpc_paths = mpc_paths

    def set_init_var(self, horizon, nr_samples):
        self.mpc_paths = np.zeros((horizon, nr_samples, self.state.shape[0]))
        #(self.horizon, self.N, self.cur_state.shape[0])


        



    
