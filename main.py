from util import *
from cdog import *

import re
from collections import defaultdict
import pandas as pd
import time

if __name__ == '__main__':
    data_path='./benchmarks/BenchmarkVirtualVariousNVar000/batch209'
    camera_pose_path="./data/poses"

    # camera poses, stack of poses of all views. Intrinsic Ks, rotation Rs, translation Ts
    Ks, Rs, Ts=load_camera_poses(camera_pose_path)

    points_2D_view,points_2D_group,points_3D,node_set=load_data(data_path,0)

    # the third column is group id, I removed it in this demo version. 
    # You can comment out the following line for group id visualization, will need minor change in the code (estimate 2 minuts)
    points_2D_view = [arr[:, :2] for arr in points_2D_view]

    tracker = EpipolarTracker(10)
    matched_groups,variances = tracker.epipolar_matching(0.5,10,node_set,points_2D_view, Rs, Ts, Ks)
    decoded_groups,decoded_variances=decode_results(matched_groups,points_2D_view,variances)
    decoded_groups=np.array(decoded_groups)
    
    print("association results")
    for i,tmp in enumerate(decoded_groups):
        print('new group')
        print(tmp)
