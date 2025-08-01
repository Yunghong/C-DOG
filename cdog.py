# Epipolar matching My Method Bidirectional
import numpy as np
from itertools import combinations
from collections import deque, defaultdict
from typing import List, Tuple, Set, Optional, Dict, Tuple
import sys
import ast
import math
from collections import Counter
import os

from nodeSystem import NodeSystem
from IQR import OutlierDetector

class EpipolarTracker:
    """Python implementation of the EpipolarTrackor class."""
    
    def __init__(self, view_number: int, encode_val: int = 1000):
        self.view_number = view_number
        self.encode_val = encode_val
        self.max_index_bean = 0
    
    @staticmethod
    def get_distance(x1: float, y1: float, a: float, b: float, c: float) -> float:
        """
        Compute the distance from a point to a line.
        
        Args:
            x1, y1: Point coordinates
            a, b, c: Line equation coefficients (ax + by + c = 0)
            
        Returns:
            Distance from point to line
        """
        return abs(a * x1 + b * y1 + c) / math.sqrt(a * a + b * b)
    
    @staticmethod
    def get_fundamental_matrix(Ra: np.ndarray, Rb: np.ndarray, ta: np.ndarray, 
                             tb: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Compute the fundamental matrix between two views.
        
        Args:
            Ra, Rb: Rotation matrices (3x3)
            ta, tb: Translation vectors (3x1)
            Ks: Camera intrinsic matrix (3x3)
            
        Returns:
            Fundamental matrix (3x3)
        """
        # Compute translation vector T
        T = K @ tb - K @ Rb @ np.linalg.inv(Ra) @ ta

        
        # Create skew-symmetric matrix from T
        T_cross = np.array([
            [0, -T[2, 0], T[1, 0]],
            [T[2, 0], 0, -T[0, 0]],
            [-T[1, 0], T[0, 0], 0]
        ])
        
        # Compute fundamental matrix
        F = T_cross @ K @ Rb @ np.linalg.inv(Ra) @ np.linalg.inv(K)
        return F
    
    @staticmethod
    def get_line_points(l: np.ndarray, width: int = 640) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get two points on the epipolar line.
        
        Args:
            l: Line coefficients (1x3 array)
            width: Image width for second point
            
        Returns:
            Two points on the line as ((x1, y1), (x2, y2))
        """
        x1 = 0
        y1 = int(-(l[0] * x1 + l[2]) / l[1])
        
        x2 = width
        y2 = int(-(l[0] * x2 + l[2]) / l[1])
        
        return (x1, y1), (x2, y2)

    def second_association(self,groups,missed_node_groups,beans_mv,matched_groups,Rs,Ts,K,distThr):
        for missed_node_group in missed_node_groups:
            for i,missed_node in enumerate(missed_node_group):
                if missed_node != -1:
                    # print('groups')
                    # print(groups)
                    # print('end')
                    unsigned_point=beans_mv[i][missed_node]
                    distanceGA=[]
                    for group in groups:
                        distances=[]

                        existing_views=[]
                        for j,node in enumerate(group):
                            if node!=-1:
                                existing_views.append(j)

                        if i in existing_views:
                            distanceGA.append(9999)
                            continue
                            
                        for j,node in enumerate(group):
                            if node!=-1 and not i in existing_views:
                                point=beans_mv[j][node]
                                                                
                                # Get fundamental matrix from view i to view j
                                F_ij = self.get_fundamental_matrix(Rs[i], Rs[j], Ts[i], Ts[j], K)
                                
                                # Get epipolar line in view i corresponding to point in view j
                                point_j = np.array([[point[0], point[1], 1.0]])
                                l_j2i = point_j @ F_ij  # Epipolar line in view i
                                
                                distance=self.get_distance(
                                        unsigned_point[0], unsigned_point[1],
                                        l_j2i[0, 0], l_j2i[0, 1], l_j2i[0, 2]
                                    )
                                # distances.append(math.exp(distance-distThr))
                                if(distance<distThr):
                                    distances.append(distance)
                                else:
                                    distances.append(distance*3)
                        if(len(distances)>0):
                            distanceGA.append(np.mean(distances))
                        else:
                            distanceGA.append(9999)

                    try:
                        min_val = min(distanceGA)
                        min_idx = distanceGA.index(min_val)
    
                        # if(min_val<distThr*1.5):
                        if(min_val<distThr*0.5):
                            groups[min_idx][i]=missed_node

                            for j,node in enumerate(groups[min_idx]):
                                if(j!=i and node!=-1):
                                    positional_encode1 = j * self.encode_val + node
                                    positional_encode2 = i * self.encode_val + missed_node
                                    matched_groups.add_connection(positional_encode1, positional_encode2,weight=min_val)
                                    matched_groups.add_connection(positional_encode2, positional_encode1,weight=min_val)
                        else:
                            matched_groups.add_node(i * self.encode_val + missed_node)
                    except:
                        print('no associated group in the second association')
                        
        

    def kick_outliers(self,Rs,Ts,Ks,matched_groups,groupsOutput,beans_mv):
        # decoded_groups,decoded_variances=decode_results(matched_groups,points_2D_view,variances)
        # decoded_groups=np.array(decoded_groups)
        detector = OutlierDetector(Rs, Ts, Ks)

        groups = []
        
        for group_id, nodes in groupsOutput.items():
        # for group in groupsOutput:
            if len(nodes) > 1:
                group_tmp = [-1] * self.view_number
                
                for node_id in nodes:
                    node = matched_groups.all_nodes[node_id]
                    # Decode: which view and which bean index
                    view_idx = node.id // self.encode_val
                    points_idx = node.id % self.encode_val
                    group_tmp[view_idx] = points_idx
                
                groups.append(group_tmp)
        
        for group in groups:
            x=np.ones((len(Rs),2))*-1

            for i,ptIdx in enumerate(group):
                if(ptIdx!=-1):
                    x[i]=beans_mv[i][ptIdx]
                
            
            while True:
                outlier_indices = detector.process(x)  # x is (N, 2) or (N, 3)
                # print(i,outlier_indices)
            
                if len(outlier_indices)==0:
                    break
                else:
                    for idx in outlier_indices:                        
                        x[idx]=np.array([-1,-1])
                        node=idx*self.encode_val+group[idx]

                        group[idx]=-1


                        matched_groups.delete_node(node)
                        
        
    def epipolar_matching(self, strengthThr, distThr, node_set, beans_mv: List[List[np.ndarray]], 
                         Rs: List[np.ndarray], Ts: List[np.ndarray], 
                         Ks: np.ndarray) -> List[List[int]]:
        """
        Perform epipolar matching between multiple views.
        
        Args:
            beans_mv: List of lists of bean centers (2D points) for each view
            Rs: List of rotation matrices for each view
            ts: List of translation vectors for each view
            K: Camera intrinsic matrix
            
        Returns:
            List of matched groups, where each group contains bean indices for each view
        """
        # Find the view with the most beans
        max_bean_n = 0
        self.max_index_bean = 0
        
        for i, beans in enumerate(beans_mv):
            if max_bean_n < len(beans):
                max_bean_n = len(beans)
                self.max_index_bean = i
        
        # Initialize NodeSystem for matching
        matched_groups = NodeSystem()
        
        # Create points matrix: [bean_index][view_index] -> (x, y) or (-1, -1) if no point
        pts = []
        for i in range(max_bean_n):
            view_points = []
            for j in range(self.view_number):
                if i < len(beans_mv[j]):
                    # beans_mv[j][i] is expected to be a numpy array with bean center coordinates
                    point = beans_mv[j][i]  # Should be [x, y] or similar
                    view_points.append((float(point[0]), float(point[1])))
                else:
                    view_points.append((-1.0, -1.0))
            pts.append(view_points)
        
        # Initialize scores tensor: [view_i][view_j][bean_k] -> matched_bean_index or -1
        scores = np.full((self.view_number, self.view_number, max_bean_n), -1.0)
        
        # Compute epipolar matching scores
        for i in range(self.view_number):
            for j in range(self.view_number):
                if i != j:
                    for k in range(max_bean_n):
                        # Check if point exists in view j
                        if pts[k][j][0] == -1:
                            continue
                        
                        # Get fundamental matrix from view i to view j
                        F_ij = self.get_fundamental_matrix(Rs[i], Rs[j], Ts[i], Ts[j], Ks[0])
                        
                        # Get epipolar line in view i corresponding to point in view j
                        point_j = np.array([[pts[k][j][0], pts[k][j][1], 1.0]])
                        l_j2i = point_j @ F_ij  # Epipolar line in view i
                        
                        # Compute distances from all points in view i to this epipolar line
                        distances = [1024.0] * max_bean_n  # Initialize with large values
                        
                        for m in range(max_bean_n):
                            if pts[m][i][0] != -1:  # Point exists in view i
                                distances[m] = self.get_distance(
                                    pts[m][i][0], pts[m][i][1],
                                    l_j2i[0, 0], l_j2i[0, 1], l_j2i[0, 2]
                                )

                        
                        # Find the minimum distance and corresponding point
                        min_score = min(distances)
                        
                        # Only accept matches with distance less than threshold
                        if min_score < distThr:
                            min_index = distances.index(min_score)

                            positional_encode1 = j * self.encode_val + k
                            positional_encode2 = i * self.encode_val + min_index
                            matched_groups.add_connection(positional_encode1, positional_encode2,weight=min_score)


        
        # Split groups based on connection strength threshold
        # Lower threshold means fewer, more confident groups
        removed_count = matched_groups.remove_connections_by_weight_threshold(distThr)
        matched_groups.detect_and_split_groups(strengthThr)

        groupsOutput=matched_groups.get_group_summary()

        self.kick_outliers(Rs,Ts,Ks,matched_groups,groupsOutput,beans_mv)

        groupsOutput=matched_groups.get_group_summary()
        groups = []

        missed_node_groups=[]

        valid_nodes=[]
        for group_id, nodes in groupsOutput.items():
            if len(nodes) > 1:
                group_tmp = [-1] * self.view_number
                
                for node_id in nodes:
                    node = matched_groups.all_nodes[node_id]
                    view_idx = node.id // self.encode_val
                    points_idx = node.id % self.encode_val
                    group_tmp[view_idx] = points_idx

                    valid_nodes.append((view_idx,points_idx))
                
                groups.append(group_tmp)

            elif(len(nodes) == 1):
                group_tmp = [-1] * self.view_number
                
                for node_id in nodes:
                    node = matched_groups.all_nodes[node_id]
                    view_idx = node.id // self.encode_val
                    points_idx = node.id % self.encode_val
                    group_tmp[view_idx] = points_idx
                    valid_nodes.append((view_idx,points_idx))
                
                missed_node_groups.append(group_tmp)

        for node in node_set:
            if(not node in valid_nodes):
                group_tmp = [-1] * self.view_number
                group_tmp[node[0]] = node[1]
                missed_node_groups.append(group_tmp)

        self.second_association(groups,missed_node_groups,beans_mv,matched_groups,Rs,Ts,Ks[0],distThr)
        
        # Get programmatic access to variation data
        variations = matched_groups.calculate_group_connection_variations()

        groupsOutput=matched_groups.get_group_summary()
        final_groups=[]
        for group_id, nodes in groupsOutput.items():
            if len(nodes) > 1:
                group_tmp = [-1] * self.view_number
            
                for node_id in nodes:
                    node = matched_groups.all_nodes[node_id]
                    # Decode: which view and which bean index
                    view_idx = node.id // self.encode_val
                    points_idx = node.id % self.encode_val
                    group_tmp[view_idx] = points_idx
                
                final_groups.append(group_tmp)

        
        return final_groups,variations