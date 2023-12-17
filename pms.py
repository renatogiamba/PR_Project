import argparse
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import utils
import ls

if __name__ == "__main__":
    
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--kernel_threshold", action="store",
        default= 1e3, help="set the kernel threshold value")
    cli.add_argument(
        "--damping", action="store",
        default= 1, help="set the damping value"
    )
    cli.add_argument(
        "--start_num_iterations", action="store",
        default= 6, help="set the number of iterations to do in the partial least square"
    )
    cli.add_argument(
        "--num_iterations", action="store",
        default= 19, help="set the number of iterations to do in the least square"
    )
    
    args = cli.parse_args()
    
    print(f"[Reading trajectory datas...]\n")
    traj_meas,traj_gt= utils.read_trajectory('./data/trajectoy.dat');
    pose_dim= 3 
    num_poses= traj_meas.shape[1]
    print(f"[Trajectory datas correctly read]\n") 
    
    print(f"[Converting robot poses and trajectory gt to homogeneous matrices...]\n")
    XR_guess= utils.convert_robot_poses(num_poses, traj_meas)
    XR_true= utils.convert_robot_gt(num_poses,traj_gt)
    print(f"[Converted correctly]\n")
    
    print(f"[Reading landmarks positions datas...]\n")
    XL_true= utils.readLandmarksGT('./data/world.dat'); 
    num_landmarks= XL_true.shape[1]
    landmark_dim= XL_true.shape[0]
    print(f"[Positions correctly read]\n")
    
    print(f"[Reading camera datas...]\n")
    K, cam_pose, z_near, z_far, image_cols, image_rows= utils.read_camera("./data/camera.dat")
    print(f"[Camera datas correctly read]\n")
    
    Zr= utils.import_poses(num_poses, XR_guess)    
    projection_associations,Zp= utils.import_projections(num_poses, num_landmarks)
    id_landmarks = np.unique(projection_associations[1,:])
    num_landmarks = len(id_landmarks)
    print(f"[Strating landmark initialization...]\n")
    XL_guess,Zp,projection_associations,id_landmarks = utils.init_landmarks(XR_guess,Zp,projection_associations,id_landmarks, num_poses, num_landmarks, K, cam_pose)
    num_landmarks = len(id_landmarks)
    XL_guess=np.squeeze(XL_guess, axis=2)
    print(f"[Landmarks coorectly initialized]\n")
    
    print(f"[Starting preliminary landmarks optimization...]\n")
    XR_guess1,XL_guess1, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b= ls.ls_warmup(XR_guess,XL_guess,Zp,projection_associations,Zr,args.start_num_iterations,args.damping,args.kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim, K, cam_pose, z_near, image_rows, image_cols)
    print(f"[Landmarks optimization done]\n")
    
    print(f"[Starting Least Square...]\n")
    XR,XL, chi_stats_p, num_inliers_p, chi_stats_r, num_inliers_r, H, b= ls.total_ls(XR_guess1,XL_guess1,Zp,projection_associations,Zr,args.num_iterations,args.damping,args.kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim, K, cam_pose, z_near, image_rows, image_cols)
    print(f"[Least Square done]\n")
    
    utils.show_results(XR_true, XR_guess, XR, XL_true, XL_guess, XL, chi_stats_r,num_inliers_r,chi_stats_p,num_inliers_p)
    utils.evaluate_poses(XR, XR_true)