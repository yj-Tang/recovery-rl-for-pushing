import os
import sys
import numpy as np
from scipy import integrate
from general_problem_setup import Problem, Index
import utils.utils as utils

def cylinder_pushing_dynamics(x, u, params):
    pr = Problem()
    index = Index()

    # State, x = [robot_pos_w_x, robot_pos_w_y, theta_r, v_r, d_theta_r, obj_pos_w_x, obj_pos_w_y, theta_o, vx_o, vy_o, d_theta_o]
    rob_state = x[index.rob_state]    # 0, 1, 2, 3, 4
    obj_state = x[index.obj_state]    # 5, 6, 7, 8, 9, 10

    obj_pos_w_x = obj_state[index.obj_pos_w_x]
    obj_pos_w_y = obj_state[index.obj_pos_w_y]
    theta_o = obj_state[index.theta_o]
    vx_o = obj_state[index.vx_o]
    vy_o = obj_state[index.vy_o]
    d_theta_o = obj_state[index.d_theta_o]
    robot_pos_w_x = rob_state[index.robot_pos_w_x]
    robot_pos_w_y = rob_state[index.robot_pos_w_y]
    theta_r = rob_state[index.theta_r]
    v_r = rob_state[index.v_r]
    vx_r = np.cos(theta_r) * v_r
    vy_r = np.sin(theta_r) * v_r
    d_theta_r = rob_state[index.d_theta_r]

    cos_theta_r = np.cos(theta_r)
    sin_theta_r = np.sin(theta_r)
    cos_theta_o = np.cos(theta_o)
    sin_theta_o = np.sin(theta_o)

    # control 
    d_v_r = u[index.u_acc]              # 0
    d_omega_r = u[index.u_omega_acc]    # 1
    d_vx_r = cos_theta_r*d_v_r
    d_vy_r = sin_theta_r*d_v_r

    # parameters
    rob_size = pr.rob_size  
    obj_size = pr.obj_size
    obj_mass = pr.obj_mass
    obj_I_or = pr.obj_I_or
    mu_g = pr.mu_g
    grav_acc = pr.grav_acc

    # for the robot
    d_rob_state = np.array([cos_theta_r*v_r, \
                            sin_theta_r*v_r, \
                            d_theta_r, \
                            d_v_r, \
                            d_omega_r
                            ])

    # for the object
    # closed loop kinematic chains, only compensate for the position and linear velocities
    # for position compensation
    Con = cos_theta_o * (robot_pos_w_x - obj_pos_w_x) + sin_theta_o * (robot_pos_w_y - obj_pos_w_y) + rob_size + 0.5* obj_size[0]
    d_C_q = np.array([-cos_theta_o, -sin_theta_o])
    diata_pos = np.dot(d_C_q.T, 1.0 / np.dot(d_C_q, d_C_q.T) * (-Con))
    obj_pos_w_x = obj_pos_w_x + diata_pos[0]
    obj_pos_w_y = obj_pos_w_y + diata_pos[1]
    # for velocity compensation
    d_Con = cos_theta_o * (vx_r - vx_o) - sin_theta_o * d_theta_o * (robot_pos_w_x - obj_pos_w_x) \
            + sin_theta_o * (vy_r - vy_o) + cos_theta_o * d_theta_o * (robot_pos_w_y - obj_pos_w_y)
    d_dC_dq = d_C_q
    diata_d_pos = np.dot(d_dC_dq.T, 1.0 / np.dot(d_dC_dq, d_dC_dq.T) * (-d_Con))
    vx_o = vx_o + diata_d_pos[0]
    vy_o = vy_o + diata_d_pos[1]

    # calculate the friction force and moment
    inte_f = np.array([0.0, 0.0])
    inte_m = 0.0
    pos_corner_o = np.array([[obj_size[0]/2.0, obj_size[1]/2.0], \
                            [obj_size[0]/2.0, obj_size[1]/-2.0], \
                            [obj_size[0]/-2.0, obj_size[1]/-2.0], \
                            [obj_size[0]/-2.0, obj_size[1]/2.0]])   # position of the corners in the object frame
    a = vx_o/d_theta_o
    b = vy_o/d_theta_o
    for i in range(4):  # for the object with four legs
        O_x = pos_corner_o[i, 0]
        O_y = pos_corner_o[i, 1]
        sig_v_O_x = np.array([utils.sigmoid_adap(-d_theta_o*(O_y*cos_theta_o + O_x*sin_theta_o - a)), \
                        utils.sigmoid_adap(d_theta_o*(O_x*cos_theta_o - O_y*sin_theta_o + b))])
        inte_f = inte_f + sig_v_O_x

        O_sig_v_O_x = [sig_v_O_x[0] * cos_theta_o + sig_v_O_x[1] * sin_theta_o, \
                    - sig_v_O_x[0] * sin_theta_o + sig_v_O_x[1] * cos_theta_o]
        inte_m = inte_m + (O_x * O_sig_v_O_x[1] - O_y * O_sig_v_O_x[0])
    F_fg = -mu_g * obj_mass * grav_acc / 4 * np.array([inte_f[0], inte_f[1]])
    M_fg = -mu_g* obj_mass * grav_acc / 4 * inte_m 

    # initilize the calculation matrix
    S = np.zeros([4,4])
    S[0,0] = obj_mass
    S[1,1] = obj_mass
    S[2,2] = obj_I_or
    # update the DAE matrix
    S[0,3] = d_C_q[0]
    S[1,3] = d_C_q[1]
    # S[2,3] = pos_c[1]
    S[2,3] = - sin_theta_o * (robot_pos_w_x-obj_pos_w_x) + cos_theta_o * (robot_pos_w_y-obj_pos_w_y)
    S[3,0] = S[0,3]
    S[3,1] = S[1,3]
    S[3,2] = S[2,3]

    # update the right side of the DAE equations
    DAE_right = np.array([
        F_fg[0],
        F_fg[1],
        M_fg,
        - cos_theta_o*d_vx_r - sin_theta_o*d_vy_r \
        + cos_theta_o*d_theta_o*d_theta_o*(robot_pos_w_x-obj_pos_w_x) \
        + 2* sin_theta_o*d_theta_o*(vx_r-vx_o) \
        + sin_theta_o*d_theta_o*d_theta_o*(robot_pos_w_y-obj_pos_w_y) \
        - 2* cos_theta_o*d_theta_o*(vy_r-vy_o) 
    ])

    # calculate the unknowns  
    unknowns = np.dot(np.linalg.inv(S), DAE_right)
    dd_x = unknowns[0]
    dd_y = unknowns[1]
    dd_theta = unknowns[2]
    O_F_xp = unknowns[3]

    d_x = vx_o
    d_y = vy_o
    d_theta = d_theta_o
    d_obj_state = np.array([d_x, d_y, d_theta, dd_x, dd_y, dd_theta])

    d_state = np.concatenate([d_rob_state, d_obj_state], 0)

    return d_state
