import numpy as np 
from dataclasses import dataclass 


# General problem setup
@dataclass
class Problem:
    # workspace 
    # sim 
    ws_x = [-3.0, 3.0]              # m 
    ws_y = [-3.0, 3.0]              # m
    # real experiment
    # ws_x = [-3.2, 3.2]            # m 
    # ws_y = [-1.2, 1.2]            # m

    robot_maxVel = 0.5              # m/s
    robot_minVel = 0.01
    robot_maxOmega = 1.0            # rad/s 
    # robot control bound
    robot_maxAcc = 2.0              # m/s^2
    robot_minAcc = -25
    robot_maxOmegaAcc = 2.0         # rad/s^2
    robot_minOmegaAcc = -25

    # robot/object size, start and goal
    rob_size = 0.15                 # m (radius of the cylinder)
    obj_size = [0.5, 0.5]           # m (W*L)
    obj_mass = 3    # kg
    obj_I_or = 1/12 * obj_mass * \
                    (obj_size[0]**2 + obj_size[1]**2)   # moment of inertia (rotation center is 
                                                        # the center of the object) (for rectangle)
    mu_g = 0.3                     # friction coefficient of the ground-object contact surface
    mu_p = 0.2                     # friction coefficient of the robot-object contact surface
    grav_acc = 9.8                 # gravity acceleration

    # slippery bound 
    d_max = obj_size[1]/2
    phi_max = np.pi/4

    # system initial state
    robot_pos_start = [0.0, 0.0]   # m  # this value will not be used anymore
    robot_theta_start = [np.deg2rad(0.0)]
    obj_pos_start = [0.0, 0.0]
    obj_theta_start = [0.0] 
    # system target
    obj_pos_goal = [5.0, 4.]        # m
    obj_theta_goal = [np.deg2rad(0.0)]# rad
    
    # General planning settings
    dt = 0.1                # sampling time, s
    N = 10                  # horizon length
    T = N*dt                # horizon time 
    nx = 11                 # state dimension 
    nu = 2                  # control dimension
    nh = 0                  # number of inequality constraint functions
    nparam = 0              # parameter dimension
    
    # MPC cost terms weights, can be real time param
    w_px_error = 1.0
    w_py_error = 1.0
    w_theta_error = 0.0
    w_vel = 0.1
    w_omega = 0.1
    w_acc = 0.01
    w_omega_acc = 0.01

    # Feedback control gains
    k_rho = 1.0
    k_alpha = 8.0
    k_phi = -1.5
    

# General vector index
@dataclass
class Index:
    # control vector, u = [acc, omega_acc]
    u_acc = np.s_[0]        # 0
    u_omega_acc = np.s_[1]  # 1

    # state vector, x = [robot_pos_w_x, robot_pos_w_y, theta_r, v_r, d_theta_r, obj_pos_w_x, obj_pos_w_y, theta_o, vx_o, vy_o, d_theta_o]
    rob_state = np.s_[0:5]  # 0, 1, 2, 3, 4
    obj_state = np.s_[5:11] # 5, 6, 7, 8, 9, 10

    pos_r = np.s_[0: 2]     # 0, 1
    pos_o = np.s_[5: 7]     # 5, 6
    

    obj_pos_w_x = np.s_[0]  # 0
    obj_pos_w_y = np.s_[1]  # 1
    theta_o = np.s_[2]      # 2
    vx_o = np.s_[3]         # 3
    vy_o = np.s_[4]         # 4
    d_theta_o = np.s_[5]    # 5

    robot_pos_w_x = np.s_[0]    # 0
    robot_pos_w_y = np.s_[1]    # 1
    theta_r = np.s_[2]          # 2
    v_r = np.s_[3]              # 3
    d_theta_r = np.s_[4]        # 4

    # param vector, customized for different formulation