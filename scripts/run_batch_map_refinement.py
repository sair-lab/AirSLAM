import os

def MakeDir(nd):
  if not os.path.exists(nd):
    os.mkdir(nd)



dataset_idx = 0
datasets = ["euroc_norm", "uma", "oivio", "tartanair", "euroc_dark"]
dataset = datasets[dataset_idx]

workspace = "/media/code/ubuntu_files/airvio/catkin_ws"

if "euroc_norm" in dataset:
  saving_root = "/media/data/datasets/euroc/results/airvio/test"
  launch_file = "mr_euroc.launch"
elif "oivio" in dataset:
  saving_root = "/media/data/datasets/oivio/results/air_slam"
  launch_file = "mr_oivio.launch"
elif "tartanair" in dataset:
  saving_root = "/media/code/ubuntu_files/airvio/experiments/results/tartanair/abandonedfactory"
  launch_file = "mr_tartanair.launch"
if "euroc_dark" in dataset:
  saving_root = "/media/data/datasets/euroc/dark_euroc/results/air_slam"
  launch_file = "mr_euroc.launch"
else:
  print("{} is not support".format(dataset_idx))



map_root = os.path.join(saving_root, "maps")
sequences = os.listdir(map_root)
for sequence in sequences:
  seq_root = os.path.join(map_root, sequence)
  os.system("cd {} & roslaunch air_slam {} map_root:={} breakpoint:=0 visualization:=false".format(workspace, launch_file, seq_root))
