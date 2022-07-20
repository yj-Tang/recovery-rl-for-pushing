# for simulation
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate

def integ_func_test(theta, vec_o2rc_x, vec_o2rc_y, x, y): # wrong
    return (x-theta)+vec_o2rc_x*y -vec_o2rc_y

def integ_func_test(x, y, theta, vec_o2rc_x, vec_o2rc_y): # true
    return (x-theta)+vec_o2rc_x*y -vec_o2rc_y

def rot_M(theta):   
    # input: 
    # rotation angle
    # output: 
    # rotation matrix of theta, from local frame to world frame 
    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return M

def sigmoid_adap(x):
    x = 10*x
    sig = 1 / (1 + np.exp(-x))
    sig_adap = sig *2 - 1
    return sig_adap

def integ_func_fx(x, y, theta, vec_o2rc_x, vec_o2rc_y, d_theta):
    vec_x = - np.sin(theta) * x - np.cos(theta) * y + vec_o2rc_y
    return sigmoid_adap(vec_x * d_theta)

def integ_func_fy(x, y, theta, vec_o2rc_x, vec_o2rc_y, d_theta):
    vec_y = np.cos(theta) * x - np.sin(theta) * y - vec_o2rc_x
    return sigmoid_adap(vec_y * d_theta)

def integ_func_m(x, y, theta, vec_o2rc_x, vec_o2rc_y, d_theta):
    vec_x = - np.sin(theta) * x - np.cos(theta) * y + vec_o2rc_y
    vec_x = sigmoid_adap(vec_x * d_theta)
    vec_y = np.cos(theta) * x - np.sin(theta) * y - vec_o2rc_x
    vec_y = sigmoid_adap(vec_y * d_theta)
    vec_x_O = vec_x * np.cos(theta) + vec_y * np.sin(theta)
    vec_y_O = - vec_x * np.sin(theta) + vec_y * np.cos(theta)
    integ_m = x * vec_y_O - y * vec_x_O
    return integ_m

def integ_func_fx_coulomb(x, y, theta, vec_o2rc_x, vec_o2rc_y):
    vec_x = np.cos(theta) * x - np.sin(theta) * y - vec_o2rc_x
    vec_norm = x**2 + y**2 + vec_o2rc_x**2 + vec_o2rc_y**2 - 2*vec_o2rc_x*(np.cos(theta)*x - np.sin(theta)*y) \
               - 2*vec_o2rc_y*(np.sin(theta)*x + np.cos(theta)*y)
    vec_norm = np.sqrt(vec_norm)
    return vec_x / vec_norm

def integ_func_fy_coulomb(x, y, theta, vec_o2rc_x, vec_o2rc_y):
    vec_y = np.sin(theta) * x + np.cos(theta) * y - vec_o2rc_y
    vec_norm = x**2 + y**2 + vec_o2rc_x**2 + vec_o2rc_y**2 - 2*vec_o2rc_x*(np.cos(theta)*x - np.sin(theta)*y) \
               - 2*vec_o2rc_y*(np.sin(theta)*x + np.cos(theta)*y)
    vec_norm = np.sqrt(vec_norm)
    return vec_y / vec_norm

def integ_func_m_coulomb(x, y, theta, vec_o2rc_x, vec_o2rc_y):
    vec_x = np.cos(theta) * x - np.sin(theta) * y - vec_o2rc_x
    vec_y = np.sin(theta) * x + np.cos(theta) * y - vec_o2rc_y
    xx = np.cos(theta) * x - np.sin(theta) * y
    yy = np.sin(theta) * x + np.cos(theta) * y  
    vec_norm = x**2 + y**2 + vec_o2rc_x**2 + vec_o2rc_y**2 - 2*vec_o2rc_x*(np.cos(theta)*x - np.sin(theta)*y) \
               - 2*vec_o2rc_y*(np.sin(theta)*x + np.cos(theta)*y)
    vec_norm = np.sqrt(vec_norm)
    integ_m = ( xx*vec_x + yy*vec_y ) / vec_norm
    return integ_m

def boundary_2dInteg_rect(rect):
    # input:
    # rect = np.array([[0,4],[3,-1.],[10.,3], [7, 5]])  # the sorted corner points of the contact surface 
    # tri2 = np.array([[10,10],[50,31.5],[14,50]])
    xmin = np.amin(rect[:,0])
    ind_xmin = np.argwhere(rect[:,0] == xmin).flatten().tolist()
    xmax = np.amax(rect[:,0])
    ind_xmax = np.argwhere(rect[:,0] == xmax).flatten().tolist()

    if np.size(ind_xmin) > 2 or np.size(ind_xmax) > 2 :
        print("The integrated area is not a rectangle")
        y_down = None
        y_up = None
    elif np.size(ind_xmin) == 2 or np.size(ind_xmax) == 2:
        y_up = np.amax(rect[:,1])
        y_down = np.amin(rect[:,1])
    else:
        rect = np.roll(rect,-2*ind_xmin[0])   # the most left corner point is the first one in array

        # check xmax == np.amax(rect[2,0])  # x_max is the second point
        if rect[1,1] > rect[3,1]:
            y_up = lambda x : interp1d(rect[0:3,0],rect[0:3,1])(x)
            # print('rect[0:3,0]: ', rect[0:3,0])
            rect_ = np.roll(rect,-2)   # the most left corner point is the last one in array
            y_down = lambda x : interp1d(np.flip(rect_[1:4,0]),np.flip(rect_[1:4,1]))(x)
            # print('rect[1:4,0]: ', rect_[1:4,0])
        else:
            y_down = lambda x : interp1d(rect[0:3,0],rect[0:3,1])(x)
            rect_ = np.roll(rect,-2)   # the most left corner point is the last one in array
            y_up = lambda x : interp1d(np.flip(rect_[1:4,0]),np.flip(rect_[1:4,1]))(x)
    return xmin, xmax, y_down, y_up


if __name__ == "__main__":
    rect = np.array([[3,-1.],[10.,3],[7, 5],[0,4]])
    xmin, xmax, y_down, y_up = boundary_2dInteg_rect(rect)
    xr = 1
    yr = 2
    px = 0.1    # N_obj/A
    m = integrate.dblquad(integ_func_fx, xmin, xmax, y_down, y_up, args=(xr,yr,px))
