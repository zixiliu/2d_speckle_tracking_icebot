import remove_outlier
import numpy as np
import pdb


# img = np.zeros((480, 640))


velocity_dict = {}
# velocity_dict[(4, 2)] = (1/40.0, -1/40.0)
# velocity_dict[(2, 2)] = (-1/40.0, 1/40.0)
# velocity_dict[(2, 4)] = (-1/40.0, 2/40.0)
# velocity_dict[(3, 3)] = (-1/40.0, 1/40.0)

velocity_dict[(385, 231)] = (-0.35, 0.05)
velocity_dict[(383, 219)] = (-0.075, -0.325)
velocity_dict[(381, 216)] = (0.125, 0.35)
velocity_dict[(380, 213)] = (0.025, -0.05)
velocity_dict[(388, 254)] = (0.15, 0.0)
velocity_dict[(390, 236)] = (-0.35, 0.075)
velocity_dict[(392, 241)] = (0.1, -0.05)
velocity_dict[(400, 210)] = (0.15, 0.225)
velocity_dict[(406, 226)] = (-0.225, 0.075)
velocity_dict[(407, 213)] = (0.0, 0.375)
velocity_dict[(408, 213)] = (-0.025, 0.05)
velocity_dict[(409, 207)] = (-0.075, -0.025)
velocity_dict[(409, 233)] = (-0.25, 0.075)


def get_r(x, y): 
    pt = (x, y)
    neighbors = []
    for key in velocity_dict.keys():
        if key!= pt:
            neighbors.append(key)
    v_x, v_y = velocity_dict[(x, y)]
    r = remove_outlier.helper_get_normalized_residual(x,y, v_x, v_y, neighbors, velocity_dict)
    print(x, y,'r = ',r)

x, y = (385, 231)
get_r(x, y)

# x,y = (4, 2)
# get_r(x, y)


# x, y = (2, 2)
# get_r(x, y)

# x, y = (2, 4)
# get_r(x, y)

# x, y = (3, 3)
# get_r(x, y)


############### New: overall d 
## 4 points: 
# (4, 2, 'r = ', 0.7572529665807025)
# (2, 2, 'r = ', 0.08704979832055991)
# (2, 4, 'r = ', 0.22725234105293926)
# (3, 3, 'r = ', 0.0)

## 3 points: 
# (4, 2, 'r = ', 0.7549877949540326)
# (2, 2, 'r = ', 0.2196917449744465)
# (2, 4, 'r = ', 0.3985781609847785)

############### Old: x, y separate
## 4 point case: 
# (4, 2, 'r = ', 0.7109463867212112)
# (2, 2, 'r = ', 0.0)
# (2, 4, 'r = ', 0.2018789850499906)
# (3, 3, 'r = ', 0.0)

## 3 point case: 
# (4, 2, 'r = ', 0.9372076347970716)
# (2, 2, 'r = ', 0.7670977283411445)
# (2, 4, 'r = ', 0.6093209351972806)