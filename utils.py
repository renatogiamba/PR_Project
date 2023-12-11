import matplotlib.pyplot as plt
import numpy as np

def read_trajectory(file_path):
    traj_meas = []
    traj_gt = []
    
    with open(file_path, 'r') as fid:
        for line in fid:
            data = line.split()
            traj_meas.append([float(data[1]), float(data[2]), float(data[3])])
            traj_gt.append([float(data[4]), float(data[5]), float(data[6])])
    
    return np.transpose(traj_meas), np.transpose(traj_gt)

def read_camera(file_path):
    with open(file_path, 'r') as fid:
        lines = fid.readlines()

    cam_mat = np.loadtxt(lines[1:4])  
    cam_trans = np.loadtxt(lines[5:9])  

    z_near = float(lines[9].split(":")[1])  
    z_far = float(lines[10].split(":")[1])   
    width = float(lines[11].split(":")[1])   
    height = float(lines[12].split(":")[1])  

    return cam_mat, cam_trans, z_near, z_far, width, height

def readLandmarksGT(file_path):
    lan_gt = []
    with open(file_path, 'r') as fid:
        for line in fid:
            data = line.split()
            lan_gt.append([float(data[1]), float(data[2]), float(data[3])])
    return np.transpose(lan_gt)

def readMeasurements(i):
    i_str = '{:05d}'.format(i - 1)  
    file_path = f"./data/meas-{i_str}.dat"
    with open(file_path,'r') as fid:
        data = np.loadtxt(fid,usecols=(2,3,4),skiprows=3)
    id_land = data[:,0]
    measurements = data[:,1:]
    return np.transpose(id_land),np.transpose(measurements)
        

def plot_odometry_and_gt_and_landgt(traj_meas, traj_gt,lan_gt):
    traj_meas = np.array(traj_meas)  
    traj_gt = np.array(traj_gt)
    lan_gt = np.array(lan_gt)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(traj_meas[:, 0], traj_meas[:, 1],zs=0,zdir='z',label='Measurements')
    ax.plot(traj_gt[:, 0], traj_gt[:, 1],zs=0,zdir='z',label='Ground Truth')
    ax.legend()
    
    plt.show()
    
def v2t(v):
    c = np.cos(v[2])
    s = np.sin(v[2])
    A = np.array([[c, -s, v[0]],
                  [s,  c, v[1]],
                  [0,  0, 1]])
    return A

def flatten_matrix_by_columns(M):
    v = np.zeros([6,1])
    v[:4] = M[:2, :2].reshape([4,1])
    v[4:6] = M[:2, [2]]
    return v

def rand_perturb(XR_guess: list, num_poses: int):
    pert_deviation = 1
    pert_scale = np.eye(3) * pert_deviation
    for pose_num in range(2, num_poses):
        xr = np.random.rand(3,1) - 0.5
        dXr = v2t(np.dot(pert_scale, xr))
        XR_guess[:, :, pose_num] = np.dot(dXr, XR_guess[:, :, pose_num])
        
    return XR_guess

def convert_robot_poses(num_poses: int, traj_meas: list):
    XR_guess = np.zeros((3, 3, num_poses))
    for i in range(num_poses):
        XR_guess[:, :, i] = v2t(traj_meas[:, i])
    return XR_guess

def convert_robot_gt(num_poses: int, traj_gt: list):
    XR_true = np.zeros((3,3,num_poses))
    for i in range(num_poses):
        XR_true[:,:,i] = v2t(traj_gt[:,i])
    return XR_true
    
def import_poses(num_poses: int, XR_guess: list):
    Zr = np.zeros((3, 3, num_poses))
    for measurement_num in range(num_poses):
       Xi = XR_guess[:, :, (measurement_num-1)]
       Xj = XR_guess[:, :, (measurement_num-1) + 1]
       Zr[:, :, measurement_num] = np.dot(np.linalg.inv(Xi), Xj)
    return Zr

def import_projections(num_poses: int, num_landmarks: int,):
    Zp = np.zeros([2,num_poses * num_landmarks])
    projection_associations = np.zeros([2, num_poses * num_landmarks])
    measurement_num = 0
    for pose_num in range(1, num_poses+1):
        id_landmarks, measurements = readMeasurements(pose_num)
        for i in range(measurements.shape[1]):
            measurement_num += 1
            projection_associations[:, measurement_num-1] = np.transpose([pose_num, id_landmarks[i]+1])
            Zp[:, measurement_num-1] = measurements[:, i]

    return projection_associations[:,0:measurement_num],Zp[:,0:measurement_num]

def show_results(XR_true,XR_guess,XR,chi_stats_r,num_inliers_r,chi_stats_p,num_inliers_p):
    plt.figure(1)
    plt.figure().add_subplot()
    plt.plot(XR_true[0,:], XR_true[1,:],'o',color='b')
    plt.plot(XR_guess[0,:], XR_guess[1,:],'x',color='r')
    plt.show()
    
    plt.figure(2)
    plt.figure().add_subplot()
    plt.plot(XR_true[0, :], XR_true[1, :],'o',color='b')
    plt.plot(XR[0,:], XR[1,:],'x',color='r')
    plt.show()
    
    plt.figure(3)
    plt.subplot(2, 2, 1)
    plt.plot(chi_stats_r, 'r-', linewidth=2)
    plt.title("Chi Poses")
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.subplot(2, 2, 2)
    plt.plot(num_inliers_r, 'b-', linewidth=2)
    plt.title("# Inliers")
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.tight_layout()
    
    plt.subplot(2, 2, 3)
    plt.plot(chi_stats_p, 'r-', linewidth=2)
    plt.title("Chi Proj")
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.subplot(2, 2, 4)
    plt.plot(num_inliers_p, 'b-', linewidth=2)
    plt.title("# Inliers")
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.tight_layout()
    plt.show()