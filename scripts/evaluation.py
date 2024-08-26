#!/usr/bin/python
# -*- coding: UTF-8 -*- 

import os
import math
import numpy as np
import shutil

def read_tum_file(file_path):
  traj = []
  with open(file_path, "r") as f:
    for line in f.readlines():
      line = line.strip('\n').split(' ')
      pose = [float(l) for l in line]
      traj.append(pose)
  return traj

def MakeDir(nd):
  if not os.path.exists(nd):
    os.mkdir(nd)



# traj_gt_dir = "/media/code/ubuntu_files/airvo/experiments/traj_gt/oivio"
# saving_root = "/media/data/datasets/oivio/results/air_slam/"


# traj_gt_dir = "/media/code/ubuntu_files/airvo/experiments/traj_gt/uma"
# saving_root = "/media/data/datasets/uma/selected_seq/results/air_slam"


traj_gt_dir = "/media/code/ubuntu_files/airvo/experiments/traj_gt/euroc_cam"
saving_root = "/media/data/datasets/euroc/results/airvio/test"


# traj_gt_dir = "/media/bssd/datasets/tartanair/ground_truth/abandonedfactory"
# saving_root= "/media/code/ubuntu_files/airvio/experiments/results/tartanair/abandonedfactory"


# traj_gt_dir = "/media/data/datasets/euroc/dark_euroc/ground_truth"
# saving_root = "/media/data/datasets/euroc/dark_euroc/results/air_slam"



version = 0

MakeDir(saving_root)

traj_filename = "trajectory_v" + str(version) + ".txt"
sum_save_name = "rmse_v" + str(version) + ".txt"

map_root = os.path.join(saving_root, "maps")
evo_save_root = os.path.join(saving_root, "evo")
MakeDir(evo_save_root)

tmp_root = os.path.join(evo_save_root, "tmp")
MakeDir(tmp_root)
eva_root = os.path.join(tmp_root, "evaluation")
MakeDir(eva_root)
eva_seq_root = os.path.join(tmp_root, "seq")
MakeDir(eva_seq_root)

sequences = os.listdir(map_root)
print(sequences)
for sequence in sequences:
  gt_path = os.path.join(traj_gt_dir, (sequence+".txt"))
  result_root = os.path.join(map_root, sequence)
  result_path = os.path.join(result_root, traj_filename)
  
  gt_traj = read_tum_file(gt_path)
  result_traj = read_tum_file(result_path)
  print(sequence)
  if len(gt_traj) == 0 or len(result_traj) == 0:
    tracking = "LOST"
  else:
    gt_start_time = gt_traj[0][0]
    gt_end_time = gt_traj[-1][0]

    result_start_time = result_traj[0][0]
    result_end_time = result_traj[-1][0]

    print(result_end_time)
    print(gt_end_time)
    tracking = "LOST" if abs(result_end_time-gt_end_time) > 10 else "GOOD"
  print("{} : {}".format(sequence, tracking))
  if tracking == "LOST":
    continue

  eva_seq_file = sequence + ".zip"
  eva_seq_path = os.path.join(eva_seq_root, eva_seq_file)
  print(gt_path)
  print("eva_seq_path = {}".format(eva_seq_path))
  os.system("evo_ape tum {} {} -as --save_results {}".format(gt_path, result_path, eva_seq_path))

table_file = os.path.join(evo_save_root, sum_save_name)
os.system("evo_res {}/*.zip -p --use_filenames --save_table {}".format(eva_seq_root, table_file))

shutil.rmtree(tmp_root)