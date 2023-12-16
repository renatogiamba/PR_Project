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
    
def getCameraPose(XR: list, cam_pose: float):
    X_world = np.eye(4)
    X_world[:2, :2] = XR[:2, :2]
    X_world[:2, 3] = XR[:2, 2]
    X_world = np.dot(X_world, cam_pose)
    return X_world

def directionFromImgCoordinates(img_coord, K):
    invK = np.linalg.inv(K)
    img_coord = np.pad(img_coord, (0,1), 'constant',constant_values=1)
    d = np.dot(invK, img_coord)
    d *= 1 / np.linalg.norm(d)
    return d

def getdirectionsfromlandmarkviews(poses, X_world, projections, K):
    directions= []
    points= []
    for pose in range(poses.size):
            points.append(X_world[:3, 3, int(poses[0][pose])-1])
            R = X_world[:3, :3, int(poses[0][pose])-1]
            directions.append(np.dot(R, directionFromImgCoordinates(projections[:, pose], K)))
    return np.array(points).swapaxes(0,1), np.array(directions).swapaxes(0,1)
    
    
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

def triangulate(points, directions):
    A = np.zeros((3, 3))
    B = np.zeros((3, 1))
    P = np.zeros((3, 1))
    for i in range(len(points[1])):
        a = directions[0, i]
        b = directions[1, i]
        c = directions[2, i]
        x = points[0, i]
        y = points[1, i]
        z = points[2, i]
        A[0, 0] += 1 - a*a
        A[0, 1] += -a*b
        A[0, 2] += -a*c
        A[1, 1] += 1 - b*b
        A[1, 2] += -b*c
        A[2, 2] += 1 - c*c
        B[0, 0] += (1 - a*a)*x - a*b*y - a*c*z
        B[1, 0] += -a*b*x + (1 - b*b)*y - b*c*z
        B[2, 0] += -a*c*x - b*c*y + (1 - c*c)*z
    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    P = np.linalg.lstsq(A, B)[0]
    return P

def init_landmarks(XR_guess, Zp, projection_associations, id_landmarks, num_poses, num_landmarks, pose_dim, landmark_dim, K, cam_pose, z_far, z_near):
    new_Zp = Zp.copy()
    new_projection_associations = projection_associations.copy()
    new_id_landmarks=[]
    XL_guess = []

    X_world = np.array([getCameraPose(XR_guess[:, :, i], cam_pose) for i in range(num_poses)]).transpose(1, 2, 0)
        
    new_num_landmarks = 0
    for current_landmark in range(num_landmarks):
        print(f"Landmark {current_landmark + 1} out of {num_landmarks}")
        
        idx = (projection_associations[1, :] == id_landmarks[current_landmark])
        poses = projection_associations[0, idx].reshape([1,-1])
        projections = Zp[:, idx]

        if poses.size < 2:
            new_Zp[:, idx] = np.array([['nan'], ['nan']], dtype=float)
            new_projection_associations[:, idx] = np.array([['nan'], ['nan']], dtype=float)
            continue
        points, directions = getdirectionsfromlandmarkviews(poses,X_world,projections, K)
        new_num_landmarks+= 1
        XL_guess.append(triangulate(points, directions))
        new_id_landmarks.append(id_landmarks[current_landmark])
        new_projection_associations[1, idx] = new_num_landmarks
        
    filter_Zp = ~np.all(np.isnan(new_Zp), axis=0)
    new_Zp = new_Zp[:, filter_Zp]
    filter_associations = ~np.all(np.isnan(new_projection_associations), axis=0)
    new_projection_associations = new_projection_associations[:, filter_associations]
    XL_guess=np.array(XL_guess).swapaxes(0,1)
    new_id_landmarks=np.array(new_id_landmarks)

    print(f"The filtered landmarks are:{num_landmarks - new_num_landmarks}")

    return XL_guess, new_Zp, new_projection_associations, new_id_landmarks

def show_results(XR_true, XR_guess, XR, XL_true, XL_guess, XL, chi_stats_r, num_inliers_r, chi_stats_p, num_inliers_p):
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
    plt.figure().add_subplot()
    ax = plt.axes(projection='3d')
    ax.plot3D(XL_true[0,:], XL_true[1,:],XL_true[2,:],'b^')
    ax.plot3D(XL_guess[0,:], XL_guess[1,:],XL_guess[2,:],'ro')
    plt.show()
    
    plt.figure(4)
    plt.figure().add_subplot()
    ax = plt.axes(projection='3d')
    ax.plot3D(XL_true[0,:], XL_true[1,:],XL_true[2,:],'b^')
    ax.plot3D(XL[0,:], XL[1,:],XL[2,:],'ro')
    plt.show()
    
    plt.figure(5)
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