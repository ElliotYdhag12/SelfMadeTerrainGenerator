"""
    A* Algorithm Implementation - To find the shortest path for my rivers
"""

import heapq
import math

class AStar:
    def __init__(self, map_matrix : tuple):
        self.map_grid = map_matrix

        self.open_set = []
        self.closed_set = []

    def search_path(self, start_node : tuple, goal_node : tuple):
        self.open_set.append(start_node) # adds the start node to the open set

        


class Node:
    def __init__(self, position : float, g : float, h : float):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h

        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f