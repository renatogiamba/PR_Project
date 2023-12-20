import matplotlib.pyplot as plt
import numpy as np

def read_trajectory(file_path):
    """
        Read the trajectory datas: measurements and ground truth .

        Parameters:
        ===========
        file_path (.dat): The input file containing datas.

        Returns:
        ========
        (np.array): The output measuremets.
        (np.array): The output ground truth.
        """
    traj_meas = []
    traj_gt = []
    
    with open(file_path, 'r') as fid:
        for line in fid:
            data = line.split()
            traj_meas.append([float(data[1]), float(data[2]), float(data[3])])
            traj_gt.append([float(data[4]), float(data[5]), float(data[6])])
    
    return np.transpose(traj_meas), np.transpose(traj_gt)

def read_camera(file_path):
    """
        Read the camera datas.

        Parameters:
        ===========
        file_path (.dat): The input file containing datas.

        Returns:
        ========
        (np.array): The output Camera matrix.
        (np.array): The output Camera transform.
        (float): The output z_near.
        (float): The output z_far.
        (float): The output width.
        (float): The output height.
        """
    with open(file_path, 'r') as fid:
        lines = fid.readlines()

    K = np.loadtxt(lines[1:4])  
    cam_pose = np.loadtxt(lines[5:9])  

    z_near = float(lines[9].split(":")[1])  
    z_far = float(lines[10].split(":")[1])   
    width = float(lines[11].split(":")[1])   
    height = float(lines[12].split(":")[1])  

    return K, cam_pose, z_near, z_far, width, height

def readLandmarksGT(file_path):
    """
        Read the landmark true positions datas.

        Parameters:
        ===========
        file_path (.dat): The input file containing datas.

        Returns:
        ========
        (np.array): The output landmarks true positions.
        """
    lan_gt = []
    with open(file_path, 'r') as fid:
        for line in fid:
            data = line.split()
            lan_gt.append([float(data[1]), float(data[2]), float(data[3])])
    return np.transpose(lan_gt)

def readMeasurements(i):
    """
        Read the landmarks measurements.

        Parameters:
        ===========
        file_path (.dat): The input file containing datas.

        Returns:
        ========
        (np.array): The output landmarks ids.
        (np.array): The output landmarks measurements.
        """
    i_str = '{:05d}'.format(i - 1)  
    file_path = f"./data/meas-{i_str}.dat"
    with open(file_path,'r') as fid:
        data = np.loadtxt(fid,usecols=(2,3,4),skiprows=3)
    id_land = data[:,0]
    measurements = data[:,1:]
    return np.transpose(id_land),np.transpose(measurements)
    
def getCameraPose(XR: list, cam_pose: list):
    """
        Convert poses in world coordinates.

        Parameters:
        ===========
        XR (list): The input file containing poses.
        cam_pose(list): The input file containing the camera transform

        Returns:
        ========
        (np.array): The output converted poses.
        """
    X_world = np.eye(4)
    X_world[:2, :2] = XR[:2, :2]
    X_world[:2, 3] = XR[:2, 2]
    X_world = np.dot(X_world, cam_pose)
    return X_world

def getdirectionsfromlandmarkviews(poses, X_world, projections, K):
    """
        Return directions and points from the views pointing at the current landmark pose.

        Parameters:
        ===========
        poses (list): The landmarks poses.
        X_world(list): The world coordinates of camera poses.
        projections(list): The projections of the current landmark.
        K(list): The camera matrix.

        Returns:
        ========
        (np.array): The output camera centers.
        (np.array): The output rays passing through the projections of the landmarks.
        """
    invK = np.linalg.inv(K)
    directions= []
    points= []
    for pose in range(poses.size):
            points.append(X_world[:3, 3, int(poses[0][pose])-1])
            R = X_world[:3, :3, int(poses[0][pose])-1]
            view = np.pad(projections[:, pose], (0,1), 'constant',constant_values=1)
            direction = np.dot(invK, view)
            direction *= 1 / np.linalg.norm(direction)
            directions.append(np.dot(R, direction))
    return np.array(points).swapaxes(0,1), np.array(directions).swapaxes(0,1)
    
    
def v2t(v):
    """
        Computes the homogeneous transformation matrix of the pose vector v.

        Parameters:
        ===========
        v (vector): The input pose vector.

        Returns:
        ========
        (np.array): The output homogeneous transformation matrix.
        """
    c = np.cos(v[2])
    s = np.sin(v[2])
    A = np.array([[c, -s, v[0]],
                  [s,  c, v[1]],
                  [0,  0, 1]])
    return A

def flatten_matrix_by_columns(M):
    """
        Computes the corresponding flattened matrix of the input matrix M.

        Parameters:
        ===========
        M (matrix): The input matrix.

        Returns:
        ========
        (np.array): The output flattened matrix.
        """
    v = np.zeros([6,1])
    v[:4] = M[:2, :2].reshape([4,1])
    v[4:6] = M[:2, [2]]
    return v

def convert_robot_poses(num_poses: int, traj_meas: list):
    """
        Computes the robot poses to an array of homogeneous matrices.

        Parameters:
        ===========
        num_poses (int): The input number of poses.
        traj_meas (list): The input trajectory measurements.

        Returns:
        ========
        (np.array): The output array of homogeneous matrices.
        """
    XR_guess = np.zeros((3, 3, num_poses))
    for i in range(num_poses):
        XR_guess[:, :, i] = v2t(traj_meas[:, i])
    return XR_guess

def convert_robot_gt(num_poses: int, traj_gt: list):
    """
        Computes the robot poses to an array of homogeneous matrices.

        Parameters:
        ===========
        num_poses (int): The input number of poses.
        traj_gt (list): The input trajectory ground truth.

        Returns:
        ========
        (np.array): The output array of homogeneous matrices.
        """
    XR_true = np.zeros((3,3,num_poses))
    for i in range(num_poses):
        XR_true[:,:,i] = v2t(traj_gt[:,i])
    return XR_true
    
def import_poses(num_poses: int, XR_guess: list):
    """
        Computes the pose-pose measurements.

        Parameters:
        ===========
        num_poses (int): The input number of poses.
        XR_guess (list): The input robot measurements array.

        Returns:
        ========
        (np.array): The output array of pose-pose measurements.
        """
    Zr = np.zeros((3, 3, num_poses))
    for measurement_num in range(num_poses):
       Xi = XR_guess[:, :, (measurement_num-1)]
       Xj = XR_guess[:, :, (measurement_num-1) + 1]
       Zr[:, :, measurement_num] = np.dot(np.linalg.inv(Xi), Xj)
    return Zr

def import_projections(num_poses: int, num_landmarks: int):
    """
        Computes the pose-landmark measurements and the associations.

        Parameters:
        ===========
        num_poses (int): The input number of poses.
        num_landmarks (list): The input number of landmarks.

        Returns:
        ========
        (np.array): The output array of associations.
        (np.array): The output array of pose-landmark measurements.
        """
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

def triangulate(points: list, directions: list):
    """
        Computes the pose-landmark measurements and the associations.

        Parameters:
        ===========
        points (list): The input camera centers.
        directions (list): The input rays passing through the projections of the landmarks.

        Returns:
        ========
        (np.array): The output solution of the linear system.
        """
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
        A[0, 0] += 1 - np.dot(a,a)
        A[0, 1] += -np.dot(a,b)
        A[0, 2] += -np.dot(a,c)
        A[1, 1] += 1 - np.dot(b,b)
        A[1, 2] += -np.dot(b,c)
        A[2, 2] += 1 - np.dot(c,c)
        B[0, 0] += np.dot((1 - np.dot(a,a)),x) - np.dot(np.dot(a,b),y) - np.dot(np.dot(a,c),z)
        B[1, 0] += -np.dot(np.dot(a,b),x) + np.dot((1 - np.dot(b,b)),y) - np.dot(np.dot(b,c),z)
        B[2, 0] += -np.dot(np.dot(a,c),x) - np.dot(np.dot(b,c),y) + np.dot((1 - np.dot(c,c)),z)
    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    P = np.linalg.lstsq(A, B)[0]
    return P

def init_landmarks(XR_guess, Zp, projection_associations, id_landmarks, num_poses, num_landmarks, K, cam_pose):
    """
        Computes landmarks positions and updates associations, number of landmarks and pose-landmarks array .

        Parameters:
        ===========
        XR_guess (list): The input camera centers.
        Zp (list): The input rays passing through the projections of the landmarks.
        projections_associations (list): The input landmarks projections association array.
        id_landmarks (list): The input landmarks ids array.
        num_poses (int): The input number od robot poses.
        num_landmarks (int): The input number of landmarks.
        K (list): The input camera matrix.
        cam_pose (list): The input camera transform.

        Returns:
        ========
        (np.array): The output landmarks positions array.
        (np.array): The output new pose-landmark array.
        (np.array): The output new associations array.
        (np.array): The output new landmarks ids array.
        """
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
        points, directions = getdirectionsfromlandmarkviews(poses, X_world, projections, K)
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

def evaluate_poses(XR,XR_true):
    """
        Evaluation function.

        Parameters:
        ===========
        XR (list): The input robot poses returned by least square.
        XR_true (list): The input robot poses ground truth.

        Returns:
        ========
        """
    num_poses = XR.shape[2]

    rotation_errors = []
    translation_errors = []
    
    for i in range(num_poses - 1):  
        rel_T = np.dot(np.linalg.inv(XR[:,:,i]), XR[:,:,i+1])
        rel_GT = np.dot(np.linalg.inv(XR_true[:,:,i]), XR_true[:,:,i+1])
        error_T = np.dot(np.linalg.inv(rel_T), rel_GT)
    
        rotation_error = np.arctan2(error_T[1, 0], error_T[0, 0])
        translation_error = np.sqrt(np.mean(error_T[0:2, 2]**2))
    
        rotation_errors.append(rotation_error)
        translation_errors.append(translation_error)
    
    avg_rotation_error = np.mean(rotation_errors)
    avg_translation_error = np.mean(translation_errors)
    
    print("Average Rotation Error (in radians):", avg_rotation_error)
    print("Average Translation RMSE Error:", avg_translation_error)
    
def show_results(XR_true, XR_guess, XR, XL_true, XL_guess, XL, chi_stats_r, num_inliers_r, chi_stats_p, num_inliers_p):
    fig1 = plt.figure(1)
    fig1.set_size_inches(12, 8)
    fig1.suptitle("Landmark and Poses", fontsize=16)
    
    ax3 = fig1.add_subplot(221)
    ax3.plot(XR_true[0,:],XR_true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax3.plot(XR_guess[0,:],XR_guess[1,:], 'x', color='r', markersize=3)
    ax3.axis([-10,10,-10,10])
    ax3.set_title("Robot ground truth and odometry values", fontsize=10)
    
    ax4 = fig1.add_subplot(222)
    ax4.plot(XR_true[0,:],XR_true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax4.plot(XR[0,:],XR[1,:], 'x', color='r', markersize=3)
    ax4.axis([-10,10,-10,10])
    ax4.set_title("Robot ground truth and optimization", fontsize=10)
    
    fig3 = plt.figure(3)
    fig3.set_size_inches(12, 6)
    fig3.suptitle("Landmarks (without outliers)", fontsize=16)
    
    ax7 = fig3.add_subplot(121)
    ax7.plot(XL_true[0,:],XL_true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax7.plot(XL_guess[0, :], XL_guess[1, :], 'x', color='r', markersize=3)
    ax7.set_title("Landmark ground truth and triangulation", fontsize=10)
    ax7.axis([-15,15,-15,15])
    
    ax8 = fig3.add_subplot(122)
    ax8.plot(XL_true[0,:],XL_true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax8.plot(XL[0,:],XL[1,:], 'x', color='r', markersize=3)
    ax8.axis([-15,15,-15,15])
    ax8.set_title("Landmark ground truth and optimization", fontsize=10)
    
    plt.show()