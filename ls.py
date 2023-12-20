import numpy as np
import utils
import poses
import projections

def total_ls(XR, XL, Zp, projection_associations, Zr, num_iterations, damping, kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim, K, cam_pose, z_near, image_rows, image_cols):
    chi_stats_p = np.zeros(num_iterations)
    num_inliers_p = np.zeros(num_iterations)
    chi_stats_r = np.zeros(num_iterations)
    num_inliers_r = np.zeros(num_iterations)
    
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    XR=XR.copy()
    XL=XL.copy()

    error=1e6

    for iteration in range(1,num_iterations):
        print(f"Iteration {iteration}/{num_iterations}")
        H = np.zeros([system_size, system_size])
        b = np.zeros([system_size,1])
        
        H_projections,b_projection, chi, num_inliers = projections.buildLinearSystemProjections(XR, XL, Zp, projection_associations, kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim, K, cam_pose, z_near, image_rows, image_cols)
        chi_stats_p[iteration]+=chi
        num_inliers_p[iteration]=num_inliers
        H += H_projections
        b += b_projection

       
        H_poses, b_poses, chi, num_inliers = poses.build_linear_system_poses(XR, XL, Zr, kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim)
        chi_stats_r[iteration] += chi
        num_inliers_r[iteration] = num_inliers
        H += H_poses
        b += b_poses

        H += np.eye(system_size) * damping
        dx = np.zeros([system_size,1])

        dx[pose_dim:] = -np.linalg.solve(H[pose_dim:, pose_dim:], b[pose_dim:,0]).reshape([-1,1])


        XR,XL= poses.box_plus(XR, XL, dx, num_poses, pose_dim, num_landmarks, landmark_dim)
        error=np.sum(np.absolute(dx))
        print("Error:"+str(error))

    return XR,XL,chi_stats_p,num_inliers_p,chi_stats_r,num_inliers_r,H,b

def ls_warmup(XR, XL, Zp, projection_associations, Zr, num_iterations, damping, kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim, K, cam_pose, z_near, image_rows, image_cols):
    chi_stats_p = np.zeros(num_iterations)
    num_inliers_p = np.zeros(num_iterations)
    chi_stats_r = np.zeros(num_iterations)
    num_inliers_r = np.zeros(num_iterations)
    
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    XR=XR.copy()
    XL=XL.copy()

    error=1e6

    for iteration in range(1,num_iterations):
        print(f"Iteration {iteration}/{num_iterations}")
        H = np.zeros([system_size, system_size])
        b = np.zeros([system_size,1])
        
        H_projections,b_projection, chi, num_inliers = projections.buildLinearSystemProjections(XR, XL, Zp, projection_associations, kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim, K, cam_pose, z_near, image_rows, image_cols)
        chi_stats_p[iteration]+=chi
        num_inliers_p[iteration]=num_inliers
        H += H_projections
        b += b_projection

        H += np.eye(system_size) * damping
        dx = np.zeros([system_size,1])

        dx[pose_dim*num_poses:] = -np.linalg.solve(H[pose_dim*num_poses:, pose_dim*num_poses:], b[pose_dim*num_poses:,0]).reshape([-1,1])


        XR,XL= poses.box_plus(XR, XL, dx, num_poses, pose_dim, num_landmarks, landmark_dim)
        error=np.sum(np.absolute(dx))
        print("Error:"+str(error))

    return XR,XL,chi_stats_p,num_inliers_p,chi_stats_r,num_inliers_r,H,b