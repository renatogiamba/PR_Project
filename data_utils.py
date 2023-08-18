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
    
    return traj_meas, traj_gt

def read_camera(file_path):
    with open(file_path, 'r') as fid:
        lines = fid.readlines()

    cam_mat = np.loadtxt(lines[1:3])  # Read camera matrix (3x3)
    cam_trans = np.loadtxt(lines[5:8])  # Read camera transform (4x4)

    z_near = float(lines[9].split(":")[1])  # Read z_near
    z_far = float(lines[10].split(":")[1])   # Read z_far
    width = float(lines[11].split(":")[1])   # Read width
    height = float(lines[12].split(":")[1])  # Read height

    return cam_mat, cam_trans, z_near, z_far, width, height

def readLandmarksGT(file_path):
    lan_gt = []
    with open(file_path, 'r') as fid:
        for line in fid:
            data = line.split()
            lan_gt.append([float(data[1]), float(data[2]), float(data[3])])
    return lan_gt

def plot_odometry_and_gt_and_landgt(traj_meas, traj_gt,lan_gt):
    traj_meas = np.array(traj_meas)  
    traj_gt = np.array(traj_gt)
    lan_gt = np.array(lan_gt)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(traj_meas[:, 0], traj_meas[:, 1],zs=0,zdir='z',label='Measurements')
    #ax.legend()   
    #bx = plt.figure().add_subplot(projection='3d')
    ax.plot(traj_gt[:, 0], traj_gt[:, 1],zs=0,zdir='z',label='Ground Truth')
    #bx.legend()  
    #cx = plt.figure().add_subplot(projection='3d')
    #ax.plot(lan_gt[:,0],lan_gt[:,1],zs=0,zdir='y',label='Landmark GT')
    ax.legend()
    
    plt.show()