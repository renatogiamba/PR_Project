import numpy as np
import utils 

def box_plus(XR, XL, dx, num_poses, pose_dim, num_landmarks, landmark_dim):

    for pose_index in range(num_poses):
        pose_matrix_index = pose_index * pose_dim
        dxr = dx[pose_matrix_index:pose_matrix_index + pose_dim]
        XR[:, :, pose_index] = np.dot(utils.v2t(dxr),XR[:, :, pose_index])

    for landmark_index in range(num_landmarks):
        landmark_matrix_index = num_poses * pose_dim + landmark_index * landmark_dim
        dxl = dx[landmark_matrix_index:landmark_matrix_index + landmark_dim]
        XL[:, landmark_index] += dxl
    return XR,XL

R0 = np.array([[0, -1],
               [1,  0]],dtype=object)

def pose_error_and_jacobian(Xi, Xj, Z):
    
    Ri = Xi[:2, :2]
    Rj = Xj[:2, :2]
    ti = Xi[:2, 2]
    tj = Xj[:2, 2]
    tij = tj - ti
    Ri_transposed = Ri.T
    Ji = np.zeros((6, 3))
    Jj = np.zeros((6, 3))

    Jj[4:6, :2] = Ri_transposed
    Jj[:4, 2] = np.dot(Ri_transposed, np.dot(R0, Rj)).reshape(4)
    Jj[4:6, 2] = np.dot(-Ri_transposed,np.dot(R0, tj))
    Ji = -Jj

    Z_hat = np.eye(3)
    Z_hat[:2, :2] = np.dot(Ri_transposed, Rj)
    Z_hat[:2, 2] = np.dot(Ri_transposed, tij)
    e = utils.flatten_matrix_by_columns(Z_hat - Z)
    return e, Ji, Jj

def build_linear_system_poses(XR, XL, Zr, kernel_threshold, num_poses, pose_dim, num_landmarks, landmark_dim):
     
    system_size = pose_dim * num_poses + landmark_dim * num_landmarks
    H = np.zeros((system_size, system_size))
    b = np.zeros((system_size))
    chi_tot = 0
    num_inliers = 0

    for measurement_num in range(Zr.shape[2]):
        Omega = np.eye(6)
        Omega[:3, :3] *= 1e3
        Z = Zr[:, :, measurement_num]
        Xi = XR[:, :, measurement_num]
        Xj = XR[:, :, measurement_num + 1]
        e, Ji, Jj = pose_error_and_jacobian(Xi, Xj, Z)
        chi = np.dot(e.T, np.dot(Omega, e))

        if chi > kernel_threshold:
            Omega *= np.sqrt(kernel_threshold / chi)
            chi = kernel_threshold
        else:
            num_inliers += 1

        chi_tot += chi
        pose_i_matrix_index = measurement_num * pose_dim
        pose_j_matrix_index = (measurement_num + 1) * pose_dim

        H[pose_i_matrix_index:pose_i_matrix_index + pose_dim,
          pose_i_matrix_index:pose_i_matrix_index + pose_dim] += np.dot(Ji.T, np.dot(Omega, Ji))

        H[pose_i_matrix_index:pose_i_matrix_index + pose_dim,
          pose_j_matrix_index:pose_j_matrix_index + pose_dim] += np.dot(Ji.T, np.dot(Omega, Jj))

        H[pose_j_matrix_index:pose_j_matrix_index + pose_dim,
          pose_i_matrix_index:pose_i_matrix_index + pose_dim] += np.dot(Jj.T, np.dot(Omega, Ji))

        H[pose_j_matrix_index:pose_j_matrix_index + pose_dim,
          pose_j_matrix_index:pose_j_matrix_index + pose_dim] += np.dot(Jj.T, np.dot(Omega, Jj))

        b[pose_i_matrix_index:pose_i_matrix_index + pose_dim] += np.dot(Ji.T, np.dot(Omega, e))
        b[pose_j_matrix_index:pose_j_matrix_index + pose_dim] += np.dot(Jj.T, np.dot(Omega, e))

    return H, b, chi_tot, num_inliers