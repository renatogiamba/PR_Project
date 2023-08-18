import matplotlib.pyplot as plt
import numpy as np
import data_utils

traj_meas,traj_gt=data_utils.read_trajectory('./PR_Project/data/trajectoy.dat');
pose_dim = np.array(traj_meas).shape[0] #3
num_poses = np.array(traj_meas).shape[1] #200
print(pose_dim,num_poses) # 3*num_poses(200)

lan_gt=data_utils.readLandmarksGT('./PR_Project/data/world.dat'); 
lan_dim = np.array(lan_gt).shape[0] #3
num_landmarks = np.array(lan_gt).shape[1] #1000
print(lan_dim,num_landmarks)

camera_params=data_utils.read_camera("./Pr_Project/data/camera.dat")
print(camera_params)

data_utils.plot_odometry_and_gt_and_landgt(traj_meas,traj_gt,lan_gt)