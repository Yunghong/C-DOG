from collections import deque, defaultdict
from typing import List, Tuple, Set, Optional, Dict, Tuple
import sys
import ast
import numpy as np
import math
from collections import Counter
import os

def load_matrix_from_txt(path):
    with open(path, 'r') as f:
        data = f.read()
        return np.array(ast.literal_eval(data))  # safely evaluates Python-like list

def count_valid_rows(list_of_arrays):
    counts = []
    for arr in list_of_arrays:
        # Boolean mask where the first column is not -1
        valid_rows = arr[:, 0] != -1
        counts.append(float(np.sum(valid_rows)))
    return counts

def most_common_third_value(list_of_list_of_arrays):
    result = []
    for sublist in list_of_list_of_arrays:
        # Extract the third value from each array in the sublist
        third_values = [arr[2] for arr in sublist if len(arr) == 3]
        counter = Counter(third_values)
        most_common = counter.most_common(1)
        # Ensure result is a plain Python scalar, not np scalar
        value = most_common[0][0] if most_common else None
        if isinstance(value, np.generic):
            value = value.item()  # convert numpy scalar to native Python type
        result.append(value)
    return result

def load_camera_poses(path):
    """
    Load intrinsic (K), rotation (R), and translation (T) matrices for 10 camera views.

    Args:
        path (str): Base directory containing K*.txt, R*.txt, and T*.txt files.

    Returns:
        Ks (list): List of 3x3 intrinsic matrices.
        Rs (list): List of 3x3 rotation matrices.
        Ts (list): List of 3x1 translation vectors.
    """
    Ks, Rs, Ts = [], [], []

    for i in range(10):
        K = load_matrix_from_txt(f'{path}/K{i}.txt')
        R = load_matrix_from_txt(f'{path}/R{i}.txt')
        T = load_matrix_from_txt(f'{path}/T{i}.txt').reshape(3, 1)

        Ks.append(K)
        Rs.append(R)
        Ts.append(T)

    return Ks, Rs, Ts

# Data Loader
def load_data(root, outputSize):
  
    max_view_index = 9

    points_2D_group = []
    points_2D_view = []
    points_3D = []

    node_set=[]
    
    for group_ID,group_name in enumerate(os.listdir(root)):
        group_path = os.path.join(root, group_name)
        if not os.path.isdir(group_path):
            continue
    
        views = []
        for i in range(max_view_index + 1):
            view_path = os.path.join(group_path, f"view{i}.txt")
            if os.path.exists(view_path):
                view = np.loadtxt(view_path).reshape(1, -1)
                view=np.append(view,group_ID)
                
            else:
                view = np.ones((1, 2))*-1  # Placeholder for missing view
                view=np.append(view,group_ID)
            views.append(view)
    
        views_stack = np.vstack(views)
        points_2D_view.append(views_stack)
    
        point_3d_path = os.path.join(group_path, "point_3D.txt")
        if os.path.exists(point_3d_path):
            point = np.loadtxt(point_3d_path).reshape(1, -1)
            points_3D.append(point)
        else:
            points_3D.append(np.zeros((1, 3)))  # Or skip group if preferred
    
    # Convert lists to numpy arrays
    points_2D_view = np.stack(points_2D_view)            # shape: (num_groups, 10, 2)
    points_3D = np.vstack(points_3D)   # shape: (num_groups, 3)

    # drop
    # Check bounds
    if outputSize < points_2D_view.shape[0] and outputSize>0:
        # Randomly select indices to keep
        keep_indices = np.random.choice(points_2D_view.shape[0], size=outputSize, replace=False) 

        # Subsample the array
        points_2D_view = points_2D_view[keep_indices]

        for i,group in enumerate(points_2D_view):
            for row in group:
                row[2]=i

        # print(points_2D_view)
        points_3D=points_3D[keep_indices]

    points_2D_group=points_2D_view.copy()

    #transfer module from numpy array to list of numpy that fits epipolar module
    transposed = points_2D_view.transpose(1, 0, 2)
    
    # Convert to list of 10 (8, 3) arrays
    points_2D_view = [transposed[i] for i in range(transposed.shape[0])]    
    points_2D_view = [arr[arr[:, 0] != -1] for arr in points_2D_view]

    for i,point_2D_view in enumerate(points_2D_view):
        for j,point in enumerate(point_2D_view):
            # print({i,j},point)
            node_set.append((i,j))

    return points_2D_view,points_2D_group,points_3D,node_set

def decode_results(matched_groups,points_2D,variances):
    groups_tmp=[]
    vars_Tmp=[]
    for i,group in enumerate(matched_groups):
        vars_Tmp.append(variances[i]['variance'])
        points_tmp=[]
        for j,idx in enumerate(group):
            if(idx!=-1):
                points_tmp.append(points_2D[j][idx])
            else:
                points_tmp.append(np.ones(2)*-1)
    
        groups_tmp.append(points_tmp)

    # print(variances)
    # for idx in variances:
    #     # print(variance)
    #     vars_Tmp.append(variances[idx]['variance'])

    return groups_tmp,vars_Tmp

def find_optimal_position(arrList):
    arr = np.array(arrList)

    min_val = arr.min()
    return np.unravel_index(arr.argmin(), arr.shape)
    
# def cal_GCA(matched_groups,points_2D):
#     OutputLen=float(len(matched_groups))
#     GTLen=float(len(points_2D))
#     return 1-abs(OutputLen-GTLen)/GTLen

# valid_ids = set()

def get_valid_group(decoded_groups):
    valid_ids = set()
    for subarr in decoded_groups:
        # Filter out rows where the third element is -1
        valid_rows = subarr[subarr[:, 2] != -1]
        
        if valid_rows.shape[0] <=1:
            continue
        
        third_values = valid_rows[:, 2]
        unique_ids = np.unique(third_values)

        
    
        if len(unique_ids) == 1:
            # print('unique_ids1',unique_ids)
            valid_ids.add(int(unique_ids[0]))

    return valid_ids

def extract_number(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else -1