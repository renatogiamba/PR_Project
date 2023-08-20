import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import data_utils
import geom_utils

#LOAD DATA
traj_meas,traj_gt=data_utils.read_trajectory('./data/trajectoy.dat');
pose_dim = traj_meas.shape[0] #3
num_poses = traj_meas.shape[1] #200

#Convert the robot poses to an array of homogeneous matrices
XR_guess = np.zeros((3, 3, num_poses))
for i in range(num_poses):
    XR_guess[:, :, i] = geom_utils.v2t(traj_meas[:, i])
#print(XR_guess[:,:,0])

lan_gt=data_utils.readLandmarksGT('./data/world.dat'); 

cam_mat, cam_trans, z_near, z_far, width, height=data_utils.read_camera("./data/camera.dat")

#POSE MEAS.
Zr = np.zeros((3, 3, num_poses - 1))
for measurement_num in range(num_poses - 1):
    Xi = XR_guess[:, :, measurement_num]
    Xj = XR_guess[:, :, measurement_num + 1]
    Zr[:, :, measurement_num] = np.linalg.inv(Xi) @ Xj
    
print(Zr[:,:,0])
#data_utils.plot_odometry_and_gt_and_landgt(traj_meas,traj_gt,lan_gt)

