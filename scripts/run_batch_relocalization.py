import os

def MakeDir(nd):
  if not os.path.exists(nd):
    os.mkdir(nd)

data_root = "/media/bssd/datasets/tartanair/mapping_relocalization/relocalization/abandonedfactory/sequences"
map_root = "/media/code/ubuntu_files/airvio/experiments/results/tartanair/abandonedfactory/maps"
# saving_root = "/media/code/ubuntu_files/airvio/experiments/results/tartanair/abandonedfactory/relocalization/d3_wo_sentense"
saving_root = "/media/code/ubuntu_files/airvio/experiments/results/tartanair/abandonedfactory/relocalization/d3"
workspace = "/media/code/ubuntu_files/airvio/catkin_ws"

MakeDir(saving_root)

sequences = os.listdir(map_root)
for seq in sequences:
  data_dir = os.path.join(data_root, seq)
  map_dir = os.path.join(map_root, seq)
  traj_save_path = os.path.join(saving_root, seq+".txt")

  os.system("cd {} & roslaunch air_slam reloc_tartanair.launch dataroot:={} map_root:={} traj_path:={} visualization:=false".format(workspace, data_dir, map_dir, traj_save_path))