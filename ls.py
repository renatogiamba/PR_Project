import numpy as np
import utils
import poses

def do_total_ls(XR, XL, Zp, projection_associations, Zr, num_iterations, damping, kernel_threshold,num_poses, num_landmarks, pose_dim, landmark_dim):

    chi_stats_p = np.zeros(num_iterations)
    num_inliers_p = np.zeros(num_iterations)
    chi_stats_r = np.zeros(num_iterations)
    num_inliers_r = np.zeros(num_iterations)
    
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks

    error=1e6

    for iteration in range(1,num_iterations+1):
        print(f"Iteration {iteration}/{num_iterations}")
        H = np.zeros((system_size, system_size))
        b = np.zeros((system_size))
        XR = XR.copy()

       
        H_poses, b_poses, chi, num_inliers = poses.build_linear_system_poses(XR, XL, Zr, kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim)
        chi_stats_r[iteration-1] += chi
        num_inliers_r[iteration-1] = num_inliers
        H = H_poses
        b = b_poses
    

        #print(chi_stats_r)
        #print(num_inliers_r)

        H += np.eye(system_size) * damping
        dx = np.zeros(system_size)

        dx[pose_dim:] = -np.linalg.solve(H[pose_dim:, pose_dim:], b[pose_dim:])


        XR,XL= poses.box_plus(XR, XL,dx,num_poses, pose_dim, num_landmarks, landmark_dim)
        error=np.sum(np.absolute(dx))
        print("Error:"+str(error))

    return XR,XL,chi_stats_p,num_inliers_p,chi_stats_r,num_inliers_r,H,b