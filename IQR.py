import numpy as np
from itertools import combinations
from collections import deque, defaultdict
from typing import List, Tuple, Set, Optional, Dict, Tuple
import sys
import ast
import math
from collections import Counter
import os


def triangulate_point(Rs, Ts, Ks, points_2d):
    """
    Triangulate a 3D point from two views.
    
    Args:
        Rs: list of 2 rotation matrices (2 x 3 x 3)
        Ts: list of 2 translation vectors (2 x 3,)
        Ks: list of 2 intrinsic matrices (2 x 3 x 3)
        points_2d: list of 2D points (2 x [x, y])
    
    Returns:
        P (3D point) in world coordinates (3,)
    """
    A = []

    for i in range(len(Rs)):
        R, T, K, pt = Rs[i], Ts[i], Ks[i], points_2d[i]
        P = K @ np.hstack((R, T.reshape(-1, 1)))  # 3x4 projection matrix

        x, y = pt
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    A = np.stack(A, axis=0)  # shape (4, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]

    return X[:3]


def compute_reprojection_error(P, K, R, T, point_2d):
    """
    Compute reprojection error of 3D point P to a 2D point.
    
    Args:
        P: 3D point (3,)
        K: Intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        T: Translation vector (3,)
        point_2d: observed 2D point [x, y]
        
    Returns:
        Reprojection error (float)
    """
    P_cam = R @ P + T  # in camera coordinates

    
    # print('P_cam',P_cam)
    # print(R,P,T)
    # if P_cam[2] <= 1e-6:
    #     return np.inf  # behind the camera or on plane

    p_proj = K @ P_cam

    # print('p_proj',p_proj)
    p_proj /= p_proj[2]

    # print(p_proj.T[0][:2],point_2d)
    
    return np.linalg.norm(p_proj.T[0][:2] - point_2d)

class OutlierDetector:
    def __init__(self, Rs, Ts, Ks):
        """
        Args:
            Rs: list of N (3x3) rotation matrices
            Ts: list of N (3,) translation vectors
            Ks: list of N (3x3) intrinsic matrices
        """
        self.Rs = Rs
        self.Ts = Ts
        self.Ks = Ks

    def process(self, x, alpha=1.5):
        """
        Args:
            x: array of shape (N, 2 or 3), where x[i][0] == -1 means invalid
            alpha: IQR multiplier (default 1.5)

        Returns:
            outlier_indices: list of indices in x that are detected as outliers
        """
        Rs, Ts, Ks = self.Rs, self.Ts, self.Ks

        valid_idx = [i for i, row in enumerate(x) if row[0] != -1]
        idx_pairs = list(combinations(valid_idx, 2))

        if len(valid_idx)<=2:
            return np.array([])
        
        voteError=np.zeros((len(Rs),2))
        # print(voteError)
        
        for pair in idx_pairs:
            chosen = set(pair)
            not_chosen = [i for i in valid_idx if i not in chosen]
            # print(f"Chosen pair: {pair}, Not chosen: {not_chosen}")
            # print(not_chosen[0])
            for pjIdx in not_chosen:
                i,j,k=pair[0],pair[1],pjIdx
                # print('choose', i,j,'bpj',k)
            
                xtmp=np.array([x[i][:2],x[j][:2]])
                # print(xtmp)
                # x=np.array([[273.88721558,  82.6109035],[196.66821694, 268.0059935]])
                Rstmp=np.array([Rs[i],Rs[j]])
                Tstmp=np.array([Ts[i],Ts[j]])
                Kstmp=np.array([Ks[i],Ks[j]])
                
                pt3D=triangulate_point(Rstmp, Tstmp, Kstmp, xtmp).reshape(-1, 1)
                
                e1=compute_reprojection_error(pt3D, Ks[i], Rs[i], Ts[i], x[i][:2])
                e2=compute_reprojection_error(pt3D, Ks[j], Rs[j], Ts[j], x[j][:2])
                e3=compute_reprojection_error(pt3D, Ks[k], Rs[k], Ts[k], x[k][:2])

                # print(i,j,k,e1,e2,e3)
        
                voteError[i][0]+=e1
                voteError[i][1]+=1
        
                voteError[j][0]+=e2
                voteError[j][1]+=1
        
                voteError[k][0]+=e3
                voteError[k][1]+=1
                # print('e')
                # print(e1,e2,e3)

        
        
        mask = voteError[:, 1] != 0
        
        voteError[mask,0] = voteError[mask][:,0]/voteError[mask][:,1]

        # print(voteError)
        if(max(voteError[:,0])<10):
            return np.array([])

        if(max(voteError[:,0])==0):
            return np.array([])
        
        # Step 1: Extract the first column
        scores = voteError[:, 0]
        
        # Step 2: Identify valid (non-zero) entries
        valid_mask = scores > 0
        valid_values = scores[valid_mask]

        valid_values_extend=valid_values.copy()
        # valid_values_extend = np.append(valid_values_extend, np.zeros(len(Rs)-len(valid_values)))
        valid_values_extend = np.append(valid_values_extend, np.array([0]))
        
        # Step 3: IQR method
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        
        alpha=1.5
        # alpha=1+(2*(len(valid_values)-1-3)/7)
        lower_bound = q1 - alpha * iqr
        upper_bound = q3 + alpha * iqr
        
        # Step 4: Reconstruct full mask
        outlier_mask = np.full_like(scores, False, dtype=bool)
        outlier_mask[valid_mask] = (valid_values < lower_bound) | (valid_values > upper_bound)
        
        # Step 5: Output indices
        outlier_indices = np.where(outlier_mask)[0]

        if(len(outlier_indices)==0 and max(valid_values)>50):
            # print(scores)
            # print(np.array([np.argmax(scores)]))
            # print(np.append(outlier_indices, np.array([np.argmax(scores)])))
        #     outlier_indices = np.append(outlier_indices, np.array([np.argmax(valid_values)]))
            outlier_indices=np.append(outlier_indices, np.array([np.argmax(scores)]))
        
        # print(x)
        # print("Outlier indices:", outlier_indices)
        # print("Outlier values:", scores[outlier_indices])
        if len(outlier_indices)==0:
            return np.array([])
        else:
            return outlier_indices
