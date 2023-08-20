close all
clear
clc


# Read odometry and ground truth
function [traj_meas,traj_gt] = readTrajectory(file_path)
  fid = fopen(file_path, 'r');
  data = textscan(fid, "%d %f %f %f %f %f %f");
  traj_meas = cell2mat(data(1,2:4))';
  traj_gt = cell2mat(data(1,5:7))';
  fclose(fid);
endfunction


# Read camera params
function [cam_mat, cam_trans, z_near, z_far, width, height] = readCamera(file_path)
  fid = fopen(file_path, 'r');
  data = textscan(fid, "%f %f %f", 3, "HeaderLines",1, "CollectOutput",1);
  cam_mat = cell2mat(data);
  data = textscan(fid, "%f %f %f %f", 4, "HeaderLines",1, "CollectOutput",1);
  cam_trans = cell2mat(data);
  data = textscan(fid, "%*s %f");
  data = cell2mat(data);
  z_near = data(1);
  z_far = data(2);
  width = data(3);
  height = data(4);
  fclose(fid);
endfunction

# Plot odometry and ground truth
function plotOdometryAndGT(traj_meas, traj_gt)
  subplot(2,1,1)
  plt=plot(traj_meas(:,1), traj_meas(:,2))
  title("Measurements")
  subplot(2,1,2)
  plt=plot(traj_gt(:,1), traj_gt(:,2))
  title("Ground Truth")
  waitfor(plt)
endfunction

#Read landmark true positions
function lan_gt = readLandmarksGT(file_path)
  fid = fopen(file_path, 'r');
  data = textscan(fid, "%*d %f %f %f");
  lan_gt = cell2mat(data)'';
  fclose(fid);
endfunction

function [id_landmarks, measurements] = readMeasurements(i)
  i = num2str(i-1,'%05.f'); % i-1 since measurements start from 0
  fid = fopen(strcat("./data/meas-",i,".dat"), 'r');
  data = textscan(fid, "%*s %*d %f %f %f", "HeaderLines",3);
  id_landmarks = cell2mat(data(1,1))';
  measurements = cell2mat(data(1,2:3))';
  fclose(fid);
endfunction

function A=v2t(v)
  c=cos(v(3));
  s=sin(v(3));
	A=[c, -s, v(1);
	   s,  c, v(2);
	   0,  0,   1];
end

# Read the trajectory (measurements & ground truth)
[traj_meas,traj_gt] = readTrajectory("./data/trajectoy.dat");
global num_poses = size(traj_meas,2)
global pose_dim = size(traj_meas,1)
#num_poses

#traj_meas(:,1)
#v2t(traj_meas(:,1))

XR_guess = zeros(3,3,num_poses);
for i = 1:num_poses
  XR_guess(:,:,i) = v2t(traj_meas(:,i));
end
#XR_guess()

# Read camera parameters
#[cam_mat, cam_trans, z_near, z_far, width, height] = readCamera("./data/camera.dat")
#lan_gt = readLandmarksGT("./data/world.dat")

#plotOdometryAndGT(traj_meas, traj_gt);

%%%%%%%%%%%%%%% POSE MEASUREMENTS %%%%%%%%%%%%%%%
Zr=zeros(3,3,num_poses-1);

for measurement_num=1:num_poses-1
  Xi=XR_guess(:,:,measurement_num);
  Xj=XR_guess(:,:,measurement_num+1);
  Zr(:,:,measurement_num)=inv(Xi)*Xj;
end

Zr(:,:,1)