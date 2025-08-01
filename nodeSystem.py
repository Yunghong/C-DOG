import numpy as np
from itertools import combinations
from collections import deque, defaultdict
from typing import List, Tuple, Set, Optional, Dict, Tuple
import sys
import ast
import math
from collections import Counter
import os
import statistics

# Enhanced Node System with Weighted Connections

class NodeG:
    """Represents a node in the graph with weighted connections to other nodes."""
    
    def __init__(self, node_id: int):
        self.id = node_id
        self.connections: Dict[int, float] = {}  # Changed to store weights
        self.encode_val=1000
    
    def add_connection(self, node_id: int, weight: float = 1.0) -> None:
        """Add a weighted connection to another node."""
        self.connections[node_id] = weight
    
    def remove_connection(self, node_id: int) -> None:
        """Remove a connection to another node."""
        self.connections.pop(node_id, None)
    
    def get_connection_count(self) -> int:
        """Get the number of connections this node has."""
        return len(self.connections)
    
    def get_connection_weight(self, node_id: int) -> Optional[float]:
        """Get the weight of connection to another node."""
        return self.connections.get(node_id)
    
    def update_connection_weight(self, node_id: int, weight: float) -> None:
        """Update the weight of an existing connection."""
        if node_id in self.connections:
            self.connections[node_id] = weight

class Group:
    """Represents a group of connected nodes."""
    
    def __init__(self):
        self.node_ids: Set[int] = set()
    
    def add_node(self, node_id: int) -> None:
        """Add a node to this group."""
        self.node_ids.add(node_id)
    
    def remove_node(self, node_id: int) -> None:
        """Remove a node from this group."""
        self.node_ids.discard(node_id)
    
    def get_node_count(self) -> int:
        """Get the number of nodes in this group."""
        return len(self.node_ids)


class NodeSystem:
    """Main system for managing nodes and their weighted groupings."""
    
    def __init__(self):
        self.all_nodes: Dict[int, NodeG] = {}
        self.groups: List[Group] = []
    
    def find_group_for_node(self, node_id: int) -> Optional[Group]:
        """Find which group contains the given node."""
        for group in self.groups:
            if node_id in group.node_ids:
                return group
        return None
    
    def merge_groups(self, group1: Group, group2: Group) -> None:
        """Merge two groups into one."""
        group1.node_ids.update(group2.node_ids)
        if group2 in self.groups:
            self.groups.remove(group2)
    
    def add_node(self, node_id: int) -> None:
        """Add a new node to the system."""
        if node_id not in self.all_nodes:
            self.all_nodes[node_id] = NodeG(node_id)
            new_group = Group()
            new_group.add_node(node_id)
            self.groups.append(new_group)
    
    def delete_node(self, node_id: int) -> None:
        """Delete a node from the system."""
        if node_id in self.all_nodes:
            # Find and remove from group
            group = self.find_group_for_node(node_id)
            if group:
                group.remove_node(node_id)
                if group.get_node_count() == 0:
                    self.groups.remove(group)
            
            # Remove connections from all other nodes
            for node in self.all_nodes.values():
                node.remove_connection(node_id)
            
            # Remove the node itself
            del self.all_nodes[node_id]
    
    def add_connection(self, id1: int, id2: int, weight: float = 1.0) -> None:
        """Add a directional weighted connection from id1 to id2."""
        # Ensure both nodes exist
        if id1 not in self.all_nodes:
            self.add_node(id1)
        if id2 not in self.all_nodes:
            self.add_node(id2)
        
        # Add directional weighted connection
        self.all_nodes[id1].add_connection(id2, weight)
        
        # Merge groups if they're different (connection makes them part of same component)
        group1 = self.find_group_for_node(id1)
        group2 = self.find_group_for_node(id2)
        
        if group1 != group2:
            self.merge_groups(group1, group2)
    
    def add_bidirectional_connection(self, id1: int, id2: int, weight1: float = 1.0, weight2: float = 1.0) -> None:
        """Add bidirectional weighted connections between two nodes (convenience method)."""
        self.add_connection(id1, id2, weight1)
        self.add_connection(id2, id1, weight2)
    
    def remove_connection(self, id1: int, id2: int) -> None:
        """Remove a directional connection from id1 to id2."""
        if id1 in self.all_nodes:
            self.all_nodes[id1].remove_connection(id2)
    
    def remove_bidirectional_connection(self, id1: int, id2: int) -> None:
        """Remove bidirectional connections between two nodes (convenience method)."""
        self.remove_connection(id1, id2)
        self.remove_connection(id2, id1)
    
    def update_connection_weight(self, id1: int, id2: int, weight: float) -> None:
        """Update the weight of a directional connection from id1 to id2."""
        if id1 in self.all_nodes:
            if id2 in self.all_nodes[id1].connections:
                self.all_nodes[id1].update_connection_weight(id2, weight)
    
    def update_bidirectional_connection_weights(self, id1: int, id2: int, weight1: float, weight2: float) -> None:
        """Update the weights of bidirectional connections (convenience method)."""
        self.update_connection_weight(id1, id2, weight1)
        self.update_connection_weight(id2, id1, weight2)
    
    def get_connection_weight(self, id1: int, id2: int) -> Optional[float]:
        """Get the weight of directional connection from id1 to id2."""
        if id1 in self.all_nodes:
            return self.all_nodes[id1].get_connection_weight(id2)
        return None
    
    def get_bidirectional_connection_weights(self, id1: int, id2: int) -> Tuple[Optional[float], Optional[float]]:
        """Get the weights of bidirectional connections between two nodes (convenience method)."""
        weight1 = self.get_connection_weight(id1, id2)
        weight2 = self.get_connection_weight(id2, id1)
        return weight1, weight2
    
    def remove_connections_by_weight_threshold(self, threshold: float) -> int:
        """
        Remove all connections where either weight is greater than the threshold.
        Returns the number of connections removed.
        """
        connections_to_remove = []
        
        # Find all connections that exceed the threshold
        for node_id, node in self.all_nodes.items():
            for connected_id, weight in node.connections.items():
                
                # if node_id < connected_id:  # Avoid checking the same pair twice
                weight1 = self.get_connection_weight(node_id, connected_id)
                weight2 = self.get_connection_weight(connected_id, node_id)

                
                if (weight1 is not None and weight1 > threshold) or (weight2 is not None and weight2 > threshold) or (weight1 is None ) or (weight2 is  None):
                    connections_to_remove.append((node_id, connected_id))
        
        # Remove both directions for identified connections
        for id1, id2 in connections_to_remove:
            self.remove_connection(id1, id2)  # Remove id1 → id2
            self.remove_connection(id2, id1)  # Remove id2 → id1
        
        # After removing connections, we need to check if groups should be split
        self._rebuild_groups_after_connection_removal()
        
        return len(connections_to_remove)
    
    def _rebuild_groups_after_connection_removal(self) -> None:
        """Rebuild groups after connections have been removed to handle disconnected components."""
        # Clear existing groups
        self.groups.clear()
        visited = set()
        
        # Use BFS to find connected components
        for node_id in self.all_nodes:
            if node_id not in visited:
                # Start a new group
                new_group = Group()
                queue = deque([node_id])
                visited.add(node_id)
                new_group.add_node(node_id)
                
                while queue:
                    current_node = queue.popleft()
                    
                    # Add all connected nodes to the same group
                    for connected_node in self.all_nodes[current_node].connections:
                        if connected_node not in visited:
                            visited.add(connected_node)
                            new_group.add_node(connected_node)
                            queue.append(connected_node)
                
                self.groups.append(new_group)
    
    #single way
    # def calculate_connection_strength(self, id1: int, id2: int) -> float:
    #     """Calculate the strength of connection between two nodes based on common connections."""
    #     if id1 not in self.all_nodes or id2 not in self.all_nodes:
    #         return 0.0
        
    #     connections1 = set(self.all_nodes[id1].connections.keys())
    #     connections2 = set(self.all_nodes[id2].connections.keys())
        
    #     common_connections = connections1.intersection(connections2)
    #     max_connections = max(len(connections1), len(connections2))

    #     print('strength',id1,id2,len(common_connections) / max_connections if max_connections > 0 else 0.0)
    #     print('connections1',connections1)
    #     print('connections2',connections2)
        
    #     return len(common_connections) / max_connections if max_connections > 0 else 0.0

    # double way
    def get_all_connected_nodes(self, node_id: int) -> Set[int]:
        """Get all nodes connected to the given node (both incoming and outgoing connections)."""
        if node_id not in self.all_nodes:
            return set()
        
        connected_nodes = set()
        
        # Add outgoing connections
        connected_nodes.update(self.all_nodes[node_id].connections.keys())
        
        # Add incoming connections (nodes that have connections to this node)
        for other_id, other_node in self.all_nodes.items():
            if node_id in other_node.connections:
                connected_nodes.add(other_id)
        
        return connected_nodes
    
    def calculate_connection_strength(self, id1: int, id2: int) -> float:
        """Calculate the strength of connection between two nodes based on common connections (bidirectional)."""
        if id1 not in self.all_nodes or id2 not in self.all_nodes:
            return 0.0
        
        # Get all connected nodes (both incoming and outgoing) for each node
        connections1 = self.get_all_connected_nodes(id1).copy()
        connections2 = self.get_all_connected_nodes(id2).copy()

        connections1.add(id1)
        connections2.add(id2)
        common_connections = connections1.intersection(connections2)
        max_connections = max(len(connections1), len(connections2))
        
        return len(common_connections) / max_connections if max_connections > 0 else 0.0
    
    # def find_weak_connections(self, group: Group, threshold: float) -> List[Tuple[int, int]]:
    #     """Find connections within a group that are below the threshold strength."""
    #     weak_connections = []
        
    #     for id1 in group.node_ids:
    #         for id2 in self.all_nodes[id1].connections:
    #             if id2 in group.node_ids and id1 < id2:  # Avoid duplicates by ensuring id1 < id2
    #                 # if self.calculate_connection_strength(id1, id2) < threshold:
    #                 if self.calculate_connection_strength(id1, id2) < threshold:
    #                     weak_connections.append((id1, id2))
        
    #     return weak_connections
    
    def split_group(self, group: Group, weak_connections: List[Tuple[int, int]]) -> None:
        """Split a group based on weak connections using BFS."""
        node_to_new_group: Dict[int, int] = {}
        new_group_id = 0
        
        # Create set of weak connections for faster lookup
        weak_set = set(weak_connections)
        
        for node_id in group.node_ids:
            if node_id not in node_to_new_group:
                # Start BFS from this node
                bfs_queue = deque([node_id])
                node_to_new_group[node_id] = new_group_id
                
                while bfs_queue:
                    current_node = bfs_queue.popleft()
                    
                    for connected_node in self.all_nodes[current_node].connections:
                        if (connected_node in group.node_ids and 
                            connected_node not in node_to_new_group):
                            
                            # Check if this connection is weak
                            min_id, max_id = min(current_node, connected_node), max(current_node, connected_node)
                            is_weak = (min_id, max_id) in weak_set
                            
                            if not is_weak:
                                node_to_new_group[connected_node] = new_group_id
                                bfs_queue.append(connected_node)
                
                new_group_id += 1
        
        # Create new groups if we found multiple components
        if new_group_id > 1:
            new_groups = [Group() for _ in range(new_group_id)]
            
            for node_id, group_id in node_to_new_group.items():
                new_groups[group_id].add_node(node_id)
            
            # Replace the original group with the first new group
            # and add the rest to the groups list
            try:
                group_index = self.groups.index(group)
                self.groups[group_index] = new_groups[0]
                self.groups.extend(new_groups[1:])
            except ValueError:
                # Group not found in list, just add all new groups
                self.groups.extend(new_groups)
    
    def detect_and_split_groups(self, threshold: float) -> None:
        """Detect and split groups based on weak connections."""
        # Work with a copy of the groups list to avoid modification during iteration
        groups_to_process = self.groups.copy()
        
        for group in groups_to_process:
            # if(group.get_node_count()<=2):
            #     continue
            weak_connections = self.find_weak_connections(group, threshold)
            if weak_connections:
                self.split_group(group, weak_connections)
                # Remove the weak connections (both directions if they exist)
                for id1, id2 in weak_connections:
                    self.remove_connection(id1, id2)
                    self.remove_connection(id2, id1)
    
    def get_group_summary(self) -> Dict[int, List[int]]:
        """
        Get a summary of which nodes are in each group.
        Returns a dictionary mapping group index to list of node IDs.
        """
        group_summary = {}
        for i, group in enumerate(self.groups):
            group_summary[i] = sorted(list(group.node_ids))
        return group_summary
    
    def print_group_summary(self) -> None:
        """Print a summary of which nodes are in each group."""
        print("Group Summary:")
        group_summary = self.get_group_summary()
        
        if not group_summary:
            print("  No groups exist")
            return
        
        for group_id, node_list in group_summary.items():
            print(f"  Group {group_id + 1}: {node_list}")
        print()

    # def calculate_group_connection_variations(self) -> Dict[int, Dict[str, float]]:
    #     """
    #     Calculate the variation of connection counts within each group.
    #     Returns a dictionary with group index as key and variation statistics as value.
    #     """
    #     import statistics
        
    #     group_variations = {}
        
    #     for group_idx, group in enumerate(self.groups):
    #         if group.get_node_count() == 0:
    #             continue
            
    #         # Get connection counts for all nodes in the group
    #         connection_counts = []
    #         for node_id in group.node_ids:
    #             total_connections = len(self.get_all_connected_nodes(node_id))+1
    #             connection_counts.append(total_connections)
            
    #         # Calculate statistics
    #         if len(connection_counts) == 0:
    #             # print('here')
    #             group_variations[group_idx] = {
    #                 'mean': connection_counts[0],
    #                 'variance': -1.0,
    #                 'std_dev': -1.0,
    #                 'min': connection_counts[0],
    #                 'max': connection_counts[0],
    #                 'connection_counts': connection_counts
    #             }
    #             # continue
    #         elif len(connection_counts) == 1:
    #             # print('here')
    #             # Only one node, no variation
    #             group_variations[group_idx] = {
    #                 'mean': connection_counts[0],
    #                 'variance': -1.0,
    #                 'std_dev': -1.0,
    #                 'min': connection_counts[0],
    #                 'max': connection_counts[0],
    #                 'connection_counts': connection_counts
    #             }
    #         else:
    #             # Multiple nodes, calculate variation
    #             mean_connections = statistics.mean(connection_counts)
    #             variance = statistics.variance(connection_counts)
    #             std_dev = statistics.stdev(connection_counts)
                
    #             group_variations[group_idx] = {
    #                 'mean': mean_connections,
    #                 'variance': variance,
    #                 'std_dev': std_dev,
    #                 'min': min(connection_counts),
    #                 'max': max(connection_counts),
    #                 'connection_counts': connection_counts
    #             }
        
    #     return group_variations

    def my_variance(self,connection_counts,mean_connections):
        N=len(connection_counts)

        vartmp=0.0

        for connections in connection_counts:
            vartmp+=((N+abs(N-connections))/N)*(connections-mean_connections)**2

        # print(vartmp)

        return vartmp/N
        

    def calculate_group_connection_variations(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate the variation of connection counts within each group.
        Returns a dictionary with group index as key and variation statistics as value.
        """
        
        
        group_variations = {}
        
        for group_idx, group in enumerate(self.groups):
            if group.get_node_count() == 0:
                continue
            
            # Get connection counts for all nodes in the group
            connection_counts = []
            for node_id in group.node_ids:
                total_connections = len(self.get_all_connected_nodes(node_id))
                connection_counts.append(total_connections)
            
            # Calculate statistics
            if len(connection_counts) == 0:
                # print('here')
                group_variations[group_idx] = {
                    'mean': connection_counts[0],
                    'variance': -1.0,
                    'std_dev': -1.0,
                    'min': connection_counts[0],
                    'max': connection_counts[0],
                    'connection_counts': connection_counts
                }
                # continue
            elif len(connection_counts) == 1:
                # print('here')
                # Only one node, no variation
                group_variations[group_idx] = {
                    'mean': connection_counts[0],
                    'variance': -1.0,
                    'std_dev': -1.0,
                    'min': connection_counts[0],
                    'max': connection_counts[0],
                    'connection_counts': connection_counts
                }
            else:
                # Multiple nodes, calculate variation
                mean_connections = statistics.mean(connection_counts)
                # variance = self.my_variance(connection_counts,mean_connections)
                variance = statistics.variance(connection_counts)
                std_dev = statistics.stdev(connection_counts)
                
                group_variations[group_idx] = {
                    'mean': mean_connections,
                    'variance': variance,
                    'std_dev': std_dev,
                    'min': min(connection_counts),
                    'max': max(connection_counts),
                    'connection_counts': connection_counts
                }
        
        return group_variations
    
    def print_group_connection_variations(self) -> None:
        """Print the variation of connection counts within each group."""
        variations = self.calculate_group_connection_variations()
        
        print("Group Connection Count Variations:")
        if not variations:
            print("  No groups with nodes exist")
            return
        
        for group_idx, stats in variations.items():
            print(f"  Group {group_idx + 1}:")
            print(f"    Connection counts: {stats['connection_counts']}")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Variance: {stats['variance']:.2f}")
            print(f"    Standard Deviation: {stats['std_dev']:.2f}")
            print(f"    Range: {stats['min']} - {stats['max']}")
        print()
        
    def print_system_info(self) -> None:
        """Print information about the current state of the system."""
        print("System Information:")
        for i, group in enumerate(self.groups):
            print(f"Group {i + 1}:")
            print(f"  Number of nodes: {group.get_node_count()}")
            for node_id in group.node_ids:
                node = self.all_nodes[node_id]
                connections_info = []
                for conn_id, weight in sorted(node.connections.items()):
                    connections_info.append(f"{conn_id}(w:{weight})")
                connections_str = " ".join(connections_info)
                print(f"  Node {node.id} connections: {connections_str}")
        print()        

    # Enhanced detect_and_split_groups method that ensures viewIdx uniqueness
    def detect_and_split_groups_with_viewidx_constraint(self, threshold: float = 0.3) -> None:
        """
        Detect and split groups to ensure each group has unique viewIdx values.
        """
        if not hasattr(self, 'encode_val'):
            self.encode_val = 1000
        
        # print("Starting group splitting with viewIdx constraint...")
        # self.print_viewidx_distribution()
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            groups_modified = False
            
            # Work with a copy to avoid modification during iteration
            groups_to_process = self.groups.copy()
            
            for group in groups_to_process:
                if group not in self.groups:  # Group might have been removed/modified
                    continue
                    
                # Check if this group needs splitting
                viewidx_to_nodes = {}
                for node_id in group.node_ids:
                    view_idx = node_id // self.encode_val
                    if view_idx not in viewidx_to_nodes:
                        viewidx_to_nodes[view_idx] = []
                    viewidx_to_nodes[view_idx].append(node_id)
                
                # If any viewIdx has multiple nodes, split the group
                needs_splitting = any(len(nodes) > 1 for nodes in viewidx_to_nodes.values())
                
                if needs_splitting:
                    # print(f"Splitting group with nodes {sorted(group.node_ids)}")
                    weak_connections = self.find_weak_connections(group, threshold)
                    
                    if weak_connections:
                        self.split_group(group, weak_connections)
                        # Remove the weak connections
                        for id1, id2 in weak_connections:
                            self.remove_connection(id1, id2)
                            self.remove_connection(id2, id1)
                        groups_modified = True
                        # print(f"Removed {len(weak_connections)} weak connections")
                    # else:
                        
                        # print(f"No weak connections found for group {sorted(group.node_ids)}")
            
            iteration += 1
            if not groups_modified:
                break
            
        #     print(f"After iteration {iteration}:")
        #     self.print_viewidx_distribution()
        
        # if iteration >= max_iterations:
        #     print(f"Warning: Reached maximum iterations ({max_iterations})")
        
        # print("Final result:")
        # self.print_viewidx_distribution()
    
    def find_weak_connections(self, group: Group, threshold: float) -> List[Tuple[int, int]]:
        """
        Find connections within a group that should be removed to ensure unique viewIdx per group.
        Uses auto-adaptive threshold and fallback strategies.
        """
        # First, add the encode_val attribute if it doesn't exist
        if not hasattr(self, 'encode_val'):
            self.encode_val = 1000  # Default value, adjust as needed
        
        # Method 1: Auto-adaptive threshold approach
        weak_connections = self._find_weak_connections_adaptive_threshold(group, threshold)
        
        # Check if adaptive threshold worked
        if self._would_create_unique_viewidx_groups(group, weak_connections):
            return weak_connections
        
        # Method 2: Fallback - Direct viewIdx-based connection removal
        # print(f"Auto-adaptive threshold didn't work for group with nodes {sorted(group.node_ids)}")
        # print("Falling back to direct viewIdx-based approach...")
        return self._find_connections_for_viewidx_separation(group)
    
    def _find_weak_connections_adaptive_threshold(self, group: Group, initial_threshold: float) -> List[Tuple[int, int]]:
        """Auto-adaptive threshold approach to find weak connections."""
        # Get viewIdx distribution in the group
        viewidx_to_nodes = {}
        for node_id in group.node_ids:
            view_idx = node_id // self.encode_val
            if view_idx not in viewidx_to_nodes:
                viewidx_to_nodes[view_idx] = []
            viewidx_to_nodes[view_idx].append(node_id)
        
        # If all viewIdx are already unique, no connections to remove
        if all(len(nodes) == 1 for nodes in viewidx_to_nodes.values()):
            return []
        
        # Try different thresholds, starting from a low value
        # thresholds_to_try = list(np.arange(0.1, 0.9, 0.05))  # 0.95 to include 0.9 due to float precision
        thresholds_to_try=[]
        thresholds_to_try.append(initial_threshold)
        # thresholds_to_try = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, initial_threshold]
        
        for thresh in thresholds_to_try:
            weak_connections = []
            
            # Find connections below threshold, prioritizing connections between same viewIdx
            for id1 in group.node_ids:
                for id2 in self.all_nodes[id1].connections:
                    if id2 in group.node_ids and id1 < id2:  # Avoid duplicates
                        connection_strength = self.calculate_connection_strength(id1, id2)
                        view_idx1 = id1 // self.encode_val
                        view_idx2 = id2 // self.encode_val
                        
                        # Prioritize breaking connections between nodes with same viewIdx
                        if view_idx1 == view_idx2:
                            # For same viewIdx, use lower threshold to more aggressively break connections
                            if connection_strength < thresh * 1.5:  # More aggressive for same viewIdx
                                weak_connections.append((id1, id2))
                        elif connection_strength < thresh:
                            weak_connections.append((id1, id2))
            
            # Test if this would create unique viewIdx groups
            if self._would_create_unique_viewidx_groups(group, weak_connections):
                # print(f"Auto-adaptive threshold {thresh:.2f} successful for group")
                return weak_connections
        
        # If no threshold worked, return connections that should definitely be broken
        return self._find_priority_connections_to_break(group)

    def _find_weak_connections_adaptive_threshold_minimize_variation(self, group: Group, initial_threshold: float) -> List[Tuple[int, int]]:
        """
        Auto-adaptive threshold approach that finds the threshold minimizing connection count variation
        while ensuring unique viewIdx per group.
        """
        # Get viewIdx distribution in the group
        viewidx_to_nodes = {}
        for node_id in group.node_ids:
            view_idx = node_id // self.encode_val
            if view_idx not in viewidx_to_nodes:
                viewidx_to_nodes[view_idx] = []
            viewidx_to_nodes[view_idx].append(node_id)
        
        # If all viewIdx are already unique, no connections to remove
        if all(len(nodes) == 1 for nodes in viewidx_to_nodes.values()):
            return []
        
        # Try different thresholds with finer granularity
        thresholds_to_try = list(np.arange(0.05, 0.95, 0.5))  # More granular search
        thresholds_to_try.append(initial_threshold)
        thresholds_to_try = sorted(set(thresholds_to_try))  # Remove duplicates and sort
        
        best_threshold = None
        best_connections = []
        best_variation = float('inf')
        
        valid_solutions = []  # Store all valid solutions for comparison
        
        for thresh in thresholds_to_try:
            weak_connections = []
            
            # Find connections below threshold, prioritizing connections between same viewIdx
            for id1 in group.node_ids:
                for id2 in self.all_nodes[id1].connections:
                    if id2 in group.node_ids and id1 < id2:  # Avoid duplicates
                        connection_strength = self.calculate_connection_strength(id1, id2)
                        view_idx1 = id1 // self.encode_val
                        view_idx2 = id2 // self.encode_val
                        
                        # Prioritize breaking connections between nodes with same viewIdx
                        if view_idx1 == view_idx2:
                            # For same viewIdx, use lower threshold to more aggressively break connections
                            if connection_strength < thresh * 1.2:  # Slightly more aggressive for same viewIdx
                                weak_connections.append((id1, id2))
                        elif connection_strength < thresh:
                            weak_connections.append((id1, id2))
            
            # Test if this would create unique viewIdx groups
            if self._would_create_unique_viewidx_groups(group, weak_connections):
                # Calculate the variation that would result from this split
                variation_score = self._calculate_split_variation(group, weak_connections)
                
                valid_solutions.append({
                    'threshold': thresh,
                    'connections': weak_connections.copy(),
                    'variation': variation_score,
                    'num_connections_removed': len(weak_connections)
                })
                
                if variation_score < best_variation:
                    best_variation = variation_score
                    best_threshold = thresh
                    best_connections = weak_connections.copy()
        
        # If we found valid solutions, return the one with minimum variation
        if valid_solutions:
            # Sort by variation score, then by number of connections removed (prefer fewer removals)
            valid_solutions.sort(key=lambda x: (x['variation'], x['num_connections_removed']))
            
            # print(f"Found {len(valid_solutions)} valid solutions for group:")
            print(f"  Best solution: threshold={valid_solutions[0]['threshold']:.3f}, "
                  f"variation={valid_solutions[0]['variation']:.3f}, "
                  f"connections_removed={valid_solutions[0]['num_connections_removed']}")
            
            return valid_solutions[0]['connections']
        
        # If no threshold worked with the adaptive approach, fall back to previous methods
        # print(f"No adaptive threshold worked for group with nodes {sorted(group.node_ids)}")
        return self._find_priority_connections_to_break(group)
    
    def _calculate_split_variation(self, original_group: Group, connections_to_remove: List[Tuple[int, int]]) -> float:
        """
        Calculate the total variation score that would result from splitting the group
        by removing the specified connections.
        """
        if not connections_to_remove:
            # If no connections to remove, calculate current variation
            return self._calculate_group_variation(original_group)
        
        # Simulate the split to get resulting components
        components = self._simulate_group_split(original_group, connections_to_remove)
        
        # Calculate total weighted variation across all resulting components
        total_variation = 0.0
        total_nodes = 0
        
        for component in components:
            if len(component) > 1:  # Only calculate variation for groups with multiple nodes
                component_variation = self._calculate_component_variation(component)
                component_size = len(component)
                
                # Weight variation by component size (larger groups contribute more to total variation)
                total_variation += component_variation * component_size
                total_nodes += component_size
            # Single-node components have zero variation, so don't contribute
        
        # Return average weighted variation
        return total_variation / max(total_nodes, 1)
    
    def _calculate_group_variation(self, group: Group) -> float:
        """Calculate variation of connection counts within a single group."""
        if group.get_node_count() <= 1:
            return 0.0
        
        connection_counts = []
        for node_id in group.node_ids:
            total_connections = len(self.get_all_connected_nodes(node_id))
            connection_counts.append(total_connections)
        
        if len(connection_counts) <= 1:
            return 0.0
        
        return statistics.variance(connection_counts)
    
    def _calculate_component_variation(self, component_nodes: Set[int]) -> float:
        """Calculate variation of connection counts within a component (set of nodes)."""
        if len(component_nodes) <= 1:
            return 0.0
        
        connection_counts = []
        for node_id in component_nodes:
            # Count connections within this component only
            connections_in_component = 0
            all_connected = self.get_all_connected_nodes(node_id)
            for connected_id in all_connected:
                if connected_id in component_nodes:
                    connections_in_component += 1
            connection_counts.append(connections_in_component)
        
        if len(connection_counts) <= 1:
            return 0.0
        
        return statistics.variance(connection_counts)
    
    def _simulate_group_split(self, original_group: Group, connections_to_remove: List[Tuple[int, int]]) -> List[Set[int]]:
        """
        Simulate splitting a group by removing specified connections and return the resulting components.
        """
        # Create temporary adjacency representation
        temp_connections = {}
        for node_id in original_group.node_ids:
            temp_connections[node_id] = set()
            # Add existing connections within the group
            for connected_id in self.all_nodes[node_id].connections:
                if connected_id in original_group.node_ids:
                    temp_connections[node_id].add(connected_id)
        
        # Remove the connections we're planning to remove
        for id1, id2 in connections_to_remove:
            temp_connections[id1].discard(id2)
            temp_connections[id2].discard(id1)
        
        # Find connected components using BFS
        visited = set()
        components = []
        
        for node_id in original_group.node_ids:
            if node_id not in visited:
                # Start new component
                component = set()
                queue = deque([node_id])
                visited.add(node_id)
                component.add(node_id)
                
                while queue:
                    current = queue.popleft()
                    for neighbor in temp_connections[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    # Enhanced method to replace the existing one in the NodeSystem class
    def find_weak_connections_minimize_variation(self, group: Group, threshold: float) -> List[Tuple[int, int]]:
        """
        Find connections within a group that should be removed to ensure unique viewIdx per group
        while minimizing the variation of connection counts within resulting groups.
        """
        # First, add the encode_val attribute if it doesn't exist
        if not hasattr(self, 'encode_val'):
            self.encode_val = 1000  # Default value, adjust as needed
        
        # Method 1: Auto-adaptive threshold approach with variation minimization
        weak_connections = self._find_weak_connections_adaptive_threshold_minimize_variation(group, threshold)
        
        # Check if adaptive threshold worked
        if self._would_create_unique_viewidx_groups(group, weak_connections):
            return weak_connections
        
        # Method 2: Fallback - Direct viewIdx-based connection removal
        # print(f"Auto-adaptive threshold didn't work for group with nodes {sorted(group.node_ids)}")
        # print("Falling back to direct viewIdx-based approach...")
        return self._find_connections_for_viewidx_separation(group)
    
    # Enhanced detect_and_split_groups method that uses the new variation-minimizing approach
    def detect_and_split_groups_minimize_variation(self, threshold: float = 0.5) -> None:
        """
        Detect and split groups to ensure unique viewIdx while minimizing connection count variation.
        """
        if not hasattr(self, 'encode_val'):
            self.encode_val = 1000
        
        # print("Starting group splitting with viewIdx constraint and variation minimization...")
        # self.print_viewidx_distribution()
        
        # Calculate initial variation
        initial_variations = self.calculate_group_connection_variations()
        initial_total_variation = sum(stats['variance'] for stats in initial_variations.values() 
                                    if stats['variance'] > 0)
        # print(f"Initial total variation: {initial_total_variation:.3f}")
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            groups_modified = False
            
            # Work with a copy to avoid modification during iteration
            groups_to_process = self.groups.copy()
            
            for group in groups_to_process:
                if group not in self.groups:  # Group might have been removed/modified
                    continue
                    
                # Check if this group needs splitting
                viewidx_to_nodes = {}
                for node_id in group.node_ids:
                    view_idx = node_id // self.encode_val
                    if view_idx not in viewidx_to_nodes:
                        viewidx_to_nodes[view_idx] = []
                    viewidx_to_nodes[view_idx].append(node_id)
                
                # If any viewIdx has multiple nodes, split the group
                needs_splitting = any(len(nodes) > 1 for nodes in viewidx_to_nodes.values())
                
                if needs_splitting:
                    # print(f"Splitting group with nodes {sorted(group.node_ids)}")
                    
                    # Use the new variation-minimizing method
                    weak_connections = self.find_weak_connections_minimize_variation(group, threshold)
                    
                    if weak_connections:
                        self.split_group(group, weak_connections)
                        # Remove the weak connections
                        for id1, id2 in weak_connections:
                            self.remove_connection(id1, id2)
                            self.remove_connection(id2, id1)
                        groups_modified = True
                        # print(f"Removed {len(weak_connections)} weak connections")
            
            iteration += 1
            if not groups_modified:
                break
            
            # print(f"After iteration {iteration}:")
            # self.print_viewidx_distribution()
        
        # Calculate final variation
        final_variations = self.calculate_group_connection_variations()
        final_total_variation = sum(stats['variance'] for stats in final_variations.values() 
                                  if stats['variance'] > 0)
        # print(f"Final total variation: {final_total_variation:.3f}")
        # print(f"Variation improvement: {initial_total_variation - final_total_variation:.3f}")
        
        # if iteration >= max_iterations:
        #     print(f"Warning: Reached maximum iterations ({max_iterations})")
        
        # print("Final result:")
        # self.print_viewidx_distribution()
        # self.print_group_connection_variations()
    
    def _find_priority_connections_to_break(self, group: Group) -> List[Tuple[int, int]]:
        """Find priority connections to break based on viewIdx conflicts."""
        weak_connections = []
        
        # Group nodes by viewIdx
        viewidx_to_nodes = {}
        for node_id in group.node_ids:
            view_idx = node_id // self.encode_val
            if view_idx not in viewidx_to_nodes:
                viewidx_to_nodes[view_idx] = []
            viewidx_to_nodes[view_idx].append(node_id)
        
        # For each viewIdx with multiple nodes, find weakest connections to break
        for view_idx, nodes in viewidx_to_nodes.items():
            if len(nodes) > 1:
                # Find all connections within this viewIdx group
                viewidx_connections = []
                for i, node1 in enumerate(nodes):
                    for j, node2 in enumerate(nodes):
                        if i < j and node2 in self.all_nodes[node1].connections:
                            strength = self.calculate_connection_strength(node1, node2)
                            viewidx_connections.append((node1, node2, strength))
                
                # Sort by strength and break the weakest connections
                viewidx_connections.sort(key=lambda x: x[2])  # Sort by strength
                
                # Break enough connections to potentially separate the nodes
                # Start with the weakest half of connections
                num_to_break = max(1, len(viewidx_connections) // 2)
                for node1, node2, _ in viewidx_connections[:num_to_break]:
                    weak_connections.append((node1, node2))
        
        return weak_connections
    
    def _would_create_unique_viewidx_groups(self, original_group: Group, connections_to_remove: List[Tuple[int, int]]) -> bool:
        """
        Simulate removing connections and check if resulting groups would have unique viewIdx.
        """
        if not connections_to_remove:
            # Check if current group already has unique viewIdx
            viewidx_set = set()
            for node_id in original_group.node_ids:
                view_idx = node_id // self.encode_val
                if view_idx in viewidx_set:
                    return False
                viewidx_set.add(view_idx)
            return True
        
        # Create a temporary adjacency representation
        temp_connections = {}
        for node_id in original_group.node_ids:
            temp_connections[node_id] = set()
            # Add existing connections within the group
            for connected_id in self.all_nodes[node_id].connections:
                if connected_id in original_group.node_ids:
                    temp_connections[node_id].add(connected_id)
        
        # Remove the connections we're planning to remove
        connections_to_remove_set = set(connections_to_remove)
        for id1, id2 in connections_to_remove:
            temp_connections[id1].discard(id2)
            temp_connections[id2].discard(id1)
        
        # Find connected components using BFS
        visited = set()
        components = []
        
        for node_id in original_group.node_ids:
            if node_id not in visited:
                # Start new component
                component = set()
                queue = deque([node_id])
                visited.add(node_id)
                component.add(node_id)
                
                while queue:
                    current = queue.popleft()
                    for neighbor in temp_connections[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        # Check if each component has unique viewIdx
        for component in components:
            viewidx_set = set()
            for node_id in component:
                view_idx = node_id // self.encode_val
                if view_idx in viewidx_set:
                    return False  # This component has duplicate viewIdx
                viewidx_set.add(view_idx)
        
        return True
    
    def _find_connections_for_viewidx_separation(self, group: Group) -> List[Tuple[int, int]]:
        """
        Fallback method: Directly identify connections that need to be removed 
        to separate nodes with the same viewIdx.
        """
        # Group nodes by viewIdx
        viewidx_to_nodes = {}
        for node_id in group.node_ids:
            view_idx = node_id // self.encode_val
            if view_idx not in viewidx_to_nodes:
                viewidx_to_nodes[view_idx] = []
            viewidx_to_nodes[view_idx].append(node_id)
        
        connections_to_remove = []
        
        # For each viewIdx with multiple nodes, use minimum cut approach
        for view_idx, nodes in viewidx_to_nodes.items():
            if len(nodes) > 1:
                # print(f"ViewIdx {view_idx} has multiple nodes: {nodes}")
                
                # Strategy: Remove connections between nodes of same viewIdx
                # and some connections to other viewIdx to ensure clean separation
                for i, node1 in enumerate(nodes):
                    for j, node2 in enumerate(nodes):
                        if i < j:
                            # Always remove direct connections between same viewIdx nodes
                            if node2 in self.all_nodes[node1].connections:
                                connections_to_remove.append((node1, node2))
                    
                    # Also remove some connections to other viewIdx nodes to help separation
                    for connected_id in self.all_nodes[node1].connections:
                        if connected_id in group.node_ids:
                            connected_view_idx = connected_id // self.encode_val
                            if connected_view_idx != view_idx:
                                # Remove connection if it's weak or if it helps separation
                                strength = self.calculate_connection_strength(node1, connected_id)
                                if strength < 0.5:  # Arbitrary threshold for cross-viewIdx connections
                                    min_id, max_id = min(node1, connected_id), max(node1, connected_id)
                                    if (min_id, max_id) not in connections_to_remove:
                                        connections_to_remove.append((min_id, max_id))
        
        return connections_to_remove
    
    # Also add a method to set the encode_val
    def set_encode_val(self, encode_val: int) -> None:
        """Set the encoding value used for viewIdx calculation."""
        self.encode_val = encode_val
    
    # Add a method to check current viewIdx distribution
    def get_viewidx_distribution(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Get the distribution of viewIdx across all groups.
        Returns: {group_index: {viewIdx: [node_ids]}}
        """
        if not hasattr(self, 'encode_val'):
            self.encode_val = 1000
        
        distribution = {}
        for group_idx, group in enumerate(self.groups):
            distribution[group_idx] = {}
            for node_id in group.node_ids:
                view_idx = node_id // self.encode_val
                if view_idx not in distribution[group_idx]:
                    distribution[group_idx][view_idx] = []
                distribution[group_idx][view_idx].append(node_id)
        
        return distribution
    
    def print_viewidx_distribution(self) -> None:
        """Print the current viewIdx distribution across groups."""
        distribution = self.get_viewidx_distribution()
        
        print("ViewIdx Distribution:")
        for group_idx, viewidx_dict in distribution.items():
            print(f"  Group {group_idx + 1}:")
            for view_idx, node_list in viewidx_dict.items():
                status = "✓" if len(node_list) == 1 else f"✗ ({len(node_list)} nodes)"
                print(f"    ViewIdx {view_idx}: {node_list} {status}")
        print()
