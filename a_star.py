"""
    A* Algorithm Implementation - To find the shortest path for my rivers
"""

import math

class AStar:
    def __init__(self, map_matrix):
        self.map_grid = map_matrix
        self.open_set = []
        self.closed_set = set()
        self.node_map = {}  # position -> Node

    def search_path(self, start_pos, goal_pos):
        self.start_node = self.get_node(start_pos)
        self.goal_node = self.get_node(goal_pos)

        self.start_node.g = 0
        self.start_node.h = self.heuristic(self.start_node, self.goal_node)
        self.start_node.f = self.start_node.h

        self.open_set.append(self.start_node)

        while self.open_set:
            # Sort open_set by f value and pop the lowest one
            self.open_set.sort()
            self.current_node = self.open_set.pop(0)

            if self.current_node == self.goal_node:
                return self.reconstruct_path(self.goal_node)

            self.closed_set.add(self.current_node)

            neighbors = self.get_neighbors(self.current_node)
            for neighbor in neighbors:
                if neighbor in self.closed_set:
                    continue

                tentative_g = self.current_node.g + 1  # Assuming cost between neighbors is 1

                if neighbor not in self.open_set:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, self.goal_node)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = self.current_node
                    self.open_set.append(neighbor)
                elif tentative_g < neighbor.g:
                    self.update_node(neighbor, tentative_g, neighbor.h)

        # No path found
        return None

    def get_node(self, position):
        if position not in self.node_map:
            self.node_map[position] = Node(position, g=math.inf, h=0)
        return self.node_map[position]

    def get_neighbors(self, node):
        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # 4-way movement
        neighbors = []

        for dx, dy in dirs:
            neighbor_pos = (node.position[0] + dx, node.position[1] + dy)

            if (0 <= neighbor_pos[0] < self.map_grid.shape[0] and
                0 <= neighbor_pos[1] < self.map_grid.shape[1]):

                if self.map_grid[neighbor_pos] != 1:  # Assuming 1 means obstacle
                    neighbors.append(self.get_node(neighbor_pos))

        return neighbors

    def heuristic(self, node, goal):
        # Manhattan distance
        return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])

    def reconstruct_path(self, end_node):
        path = []
        current = end_node

        while current:
            path.append(current.position)
            current = current.parent

        return path[::-1]  # Reverse path

    def update_node(self, node, g_cost, h_cost):
        node.g = g_cost
        node.f = g_cost + h_cost
        node.parent = self.current_node


class Node:
    def __init__(self, position, g, h):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return isinstance(other, Node) and self.position == other.position

    def __hash__(self):
        return hash(self.position)
