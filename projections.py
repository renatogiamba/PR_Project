import numpy as np
import utils

def projectionErrorAndJacobian(Xr: list, Xl: list, z: list, K: list, cam_pose: list, z_near: int, image_rows: int, image_cols: int):
    is_valid = False
    e = np.array([[0],[0]])
    Jr = np.zeros([2, 3])
    Jl = np.zeros([2, 3])

    X_robot = np.eye(4)
    X_robot[0:2, 0:2] = Xr[0:2, 0:2]
    X_robot[0:2, 3] = Xr[0:2, 2]

    iR_cam = cam_pose[0:3, 0:3].T
    it_cam = -np.dot(iR_cam, cam_pose[0:3, 3])

    iR = X_robot[0:3, 0:3].T
    it = -np.dot(iR, X_robot[0:3, 3])

    pw = np.dot(iR_cam, np.dot(iR, Xl) + it) + it_cam
    if pw[2] < z_near:
        return is_valid, e, Jr, Jl

    Jwr = np.zeros((3, 3))
    Jwr[0:3, 0:2] = -np.dot(np.dot(iR_cam, iR), np.array([[1, 0], [0, 1], [0, 0]]))
    Jwr[0:3, 2] = np.dot(np.dot(np.dot(iR_cam, iR), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])),Xl)
    Jwl = np.dot(iR_cam, iR)

    p_cam = np.dot(K, pw)
    iz = 1.0 / p_cam[2]
    z_hat = p_cam[0:2] * iz
    if (z_hat[0] < 0 or z_hat[0] > image_cols or z_hat[1] < 0 or z_hat[1] > image_rows):
        return is_valid, e, Jr, Jl

    iz2 = iz * iz
    Jp = np.array([[iz, 0, -p_cam[0] * iz2], [0, iz, -p_cam[1] * iz2]])

    e = (z_hat - z).reshape([-1,1])
    Jr = np.dot(np.dot(Jp, K), Jwr)
    Jl = np.dot(np.dot(Jp, K), Jwl)
    is_valid = True

    return is_valid, e, Jr, Jl

def buildLinearSystemProjections(XR: list, XL: list, Zp: list, associations: list, kernel_threshold: int, num_poses: int, pose_dim: int, num_landmarks: int, landmark_dim: int, K: list, cam_pose: list, z_near: int, image_rows: int, image_cols: int):
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    H = np.zeros([system_size, system_size])
    b = np.zeros([system_size,1])
    chi_tot = 0
    num_inliers = 0

    for measurement_num in range(Zp.shape[1]):
        pose_index = int(associations[0, measurement_num])
        if pose_index == np.nan:
          continue
        landmark_index = int(associations[1, measurement_num])
        z = Zp[:, measurement_num]
        Xr = XR[:, :, pose_index-1]
        Xl = XL[:,landmark_index-1]


        is_valid, e, Jr, Jl = projectionErrorAndJacobian(Xr, Xl, z, K, cam_pose, z_near, image_rows, image_cols)
        if not is_valid:
            continue

        chi = np.dot(e.T, e)
        if chi > kernel_threshold:
            e *= np.sqrt(kernel_threshold / chi)
            chi = kernel_threshold
        else:
            num_inliers += 1
        chi_tot += chi

        pose_matrix_index = (pose_index-1)*pose_dim
        landmark_matrix_index = (num_poses)*pose_dim + (landmark_index-1)*landmark_dim

        H[pose_matrix_index:pose_matrix_index+pose_dim,
          pose_matrix_index:pose_matrix_index+pose_dim] += np.dot(Jr.T, Jr)

        H[pose_matrix_index:pose_matrix_index+pose_dim,
          landmark_matrix_index:landmark_matrix_index+landmark_dim] += np.dot(Jr.T, Jl)

        H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
          landmark_matrix_index:landmark_matrix_index+landmark_dim] += np.dot(Jl.T, Jl)

        H[landmark_matrix_index:landmark_matrix_index+landmark_dim,
          pose_matrix_index:pose_matrix_index+pose_dim] += np.dot(Jl.T, Jr)

        b[pose_matrix_index:pose_matrix_index+pose_dim] += np.dot(Jr.T,e)
        b[landmark_matrix_index:landmark_matrix_index+landmark_dim] += np.dot(Jl.T,e)

    return H, b, chi_tot, num_inliers