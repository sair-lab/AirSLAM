import os

def MakeDir(nd):
  if not os.path.exists(nd):
    os.mkdir(nd)


dataset_idx = 0
datasets = ["euroc_norm", "uma", "oivio", "tartanair", "euroc_dark"]
dataset = datasets[dataset_idx]

workspace = "/media/code/ubuntu_files/airvio/catkin_ws"

if "euroc_norm" in dataset:
  dataroot = "/media/data/datasets/euroc/seq"
  saving_root = "/media/data/datasets/euroc/results/airvio/test"
  launch_file = "vo_euroc.launch"
elif "uma" in dataset:
  dataroot = "/media/data/datasets/uma/selected_seq/air_slam"
  saving_root = "/media/data/datasets/uma/selected_seq/results/air_slam"
  launch_file = "vo_uma_bumblebee.launch"
elif "oivio" in dataset:
  dataroot = "/media/data/datasets/oivio/selected_seq"
  saving_root = "/media/data/datasets/oivio/results/air_slam"
  launch_file = "vo_oivio.launch"
elif "tartanair" in dataset:
  dataroot = "/media/bssd/datasets/tartanair/euroc_style/with_time/abandonedfactory"
  saving_root = "/media/code/ubuntu_files/airvio/experiments/results/tartanair/abandonedfactory"
  launch_file = "vo_tartanair.launch"
if "euroc_dark" in dataset:
  dataroot = "/media/data/datasets/euroc/dark_euroc/sequences"
  saving_root = "/media/data/datasets/euroc/dark_euroc/results/air_slam"
  launch_file = "vo_euroc_dark.launch"
else:
  print("{} is not support".format(dataset_idx))


print(dataset_idx)
print(dataroot)

MakeDir(saving_root)
map_root = os.path.join(saving_root, "maps")
MakeDir(map_root)
sequences = os.listdir(dataroot)
# sequences = ["MH_03_medium"]
for sequence in sequences:
  seq_dataroot = os.path.join(dataroot, sequence)
  seq_save_root = os.path.join(map_root, sequence)
  MakeDir(seq_save_root)
  os.system("cd {} & roslaunch air_slam {} dataroot:={} saving_dir:={} visualization:=false".format(workspace, launch_file, seq_dataroot, seq_save_root))
