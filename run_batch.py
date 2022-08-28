import os

def MakeDir(nd):
  if os.path.exist(nd):
    os.mkdir(nd)


dataroot = "/media/xukuan/something2/ubuntu_files/data/AirSLAM/batch_test/data"
saving_root = "/media/xukuan/something2/ubuntu_files/data/AirSLAM/batch_test/saving"
workspace = "/media/xukuan/something2/ubuntu_files/project/AirSLAM/catkin_ws"
traj_gt_dir = "/media/xukuan/something2/ubuntu_files/data/AirSLAM/batch_test/gt"

traj_saving_root = os.path.join(saving_root, "traj")
MakeDir(traj_saving_root)
eva_root = os.path.join(saving_root, "evaluation")
MakeDir(eva_root)
eva_seq_root = os.path.join(saving_root, "seq")
MakeDir(eva_seq_root)
eva_sum_root = os.path.join(saving_root, "sum")
MakeDir(eva_sum_root)

os.system("cd {} & roscore & .".format(workspace))
sequences = os.listdir(dataroot)
for sequence in sequences:
  seq_dataroot = os.path.join(dataroot, sequence)
  seq_traj_file = sequence + ".txt"
  seq_traj_path = os.path.join(traj_saving_root, seq_traj_file)
  os.system("cd {} & roslaunch air_vo realsense.launch dataroot:={} traj_path:={}".format(workspace, seq_dataroot, seq_traj_path))

  

