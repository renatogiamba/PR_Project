import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import utils
import ls

#LOAD DATA
traj_meas,traj_gt= utils.read_trajectory('./data/trajectoy.dat');
pose_dim= 3 
num_poses= traj_meas.shape[1] 

XR_guess= utils.convert_robot_poses(num_poses, traj_meas)
XR_true= utils.convert_robot_gt(num_poses,traj_gt)

XL_true= utils.readLandmarksGT('./data/world.dat'); 
num_landmarks= XL_true.shape[1]
landmark_dim= XL_true.shape[0]

XL_guess= XL_true

K, cam_pose, z_near, z_far, img_width, img_height= utils.read_camera("./data/camera.dat")

Zr= utils.import_poses(num_poses, XR_guess)
    
projection_associations,Zp= utils.import_projections(num_poses, num_landmarks)

XR_guess= utils.rand_perturb(XR_guess,num_poses)

kernel_threshold= 1e3
damping= 1
num_iterations= 10
XR,XL, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b= ls.do_total_ls(XR_guess,XL_guess,Zp,projection_associations,Zr,num_iterations,damping,kernel_threshold,num_poses, num_landmarks, pose_dim, landmark_dim)

utils.show_results(XR_true,XR_guess,XR,chi_stats_r,num_inliers_r,chi_stats_p,num_inliers_p)