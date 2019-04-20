import heapq
import maze
from typing import List, Tuple, Dict
import copy

# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    states = 0
    visited = set()
    path = [maze.getStart()]
    queue = [(maze.getStart(), path)]

    while queue:
        (pos, path) = queue.pop(0)
        states += 1
        if maze.isObjective(pos[0], pos[1]):
            return path, states
        else:
            neighbors = maze.getNeighbors(pos[0], pos[1])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return path, states


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    states = 0
    visited = set()
    path = [maze.getStart()]
    stack = [(maze.getStart(), path)]

    while stack:
        (pos, path) = stack.pop()
        states += 1
        if maze.isObjective(pos[0], pos[1]):
            return path, states
        else:
            neighbors = maze.getNeighbors(pos[0], pos[1])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))

    return path, states

def mahatten(posA, posB):
    #Manhattan distance
    return abs(posA[0] - posB[0]) + abs(posA[1] - posB[1])

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    states = 0
    visited = set()
    start = maze.getStart()
    path = [start]
    pqueue = [(maze.getStart(), path)]
    objectives = maze.getObjectives()
    destination = objectives[0]
    pqueue = []
    heapq.heappush(pqueue, (mahatten(start, destination), (start, path)))

    while pqueue:
        (key, (pos, path)) = heapq.heappop(pqueue)
        states += 1
        if maze.isObjective(pos[0], pos[1]):
            return path, states
        else:
            neighbors = maze.getNeighbors(pos[0], pos[1])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(pqueue,(mahatten(neighbor, destination), (neighbor, path + [neighbor])))
    return path, states


def astar_with(heuristic):
    def astar(maze: maze.Maze):
        states = 0
        cost_so_far = {}
        objectives = frozenset(maze.getObjectives())
        start = maze.getStart()
        cost_so_far[(start, objectives)] = 0
        path = [start]
        pqueue = []
        edge_dictionary = shortest_path_all_dots(maze)
        heapq.heappush(pqueue, (heuristic(maze, start, objectives, edge_dictionary), (start, path, objectives)))

        while pqueue:
            (_, (position, path, objectives)) = heapq.heappop(pqueue)
            states = states + 1
            orig_objectives = objectives
            if maze.isObjective(position[0], position[1]):
                objectives = copy.copy(objectives)
                objectives = frozenset(set(objectives) - set([position]))
                if len(objectives) == 0:
                    return path, states
            if (position, objectives) not in cost_so_far:
                cost_so_far[(position, objectives)] = cost_so_far[(position, orig_objectives)]
            else:
                cost_so_far[(position, objectives)] = min(cost_so_far[(position, objectives)], cost_so_far[(position, orig_objectives)])
            for neighbour in maze.getNeighbors(position[0], position[1]):
                new_cost = cost_so_far[(position, objectives)] + 1
                if (neighbour, objectives) not in cost_so_far or new_cost < cost_so_far[(neighbour, objectives)]:
                    cost_so_far[(neighbour, objectives)] = new_cost
                    priority = new_cost + heuristic(maze, neighbour, objectives, edge_dictionary)
                    heapq.heappush(pqueue, (priority, (neighbour, path + [neighbour], objectives)))
        return path, states
    return astar


def astar(maze: maze.Maze):
    if len(maze.getObjectives()) < 50:
        return astar_with(heuristic_part2)(maze)
    else:
        return astar_with(heuristic_extra)(maze)


def bfs_distance(maze: maze.Maze, start, objectives):
    """Return the shortest distance from start to any of the objectives."""
    visited = set()
    queue = [(start, 0)]
    if not objectives:
        return 0
    objectives = set(objectives)
    while queue:
        (position, cost) = queue.pop(0)
        if position in objectives:
            return cost
        for neighbour in maze.getNeighbors(position[0], position[1]):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, cost + 1))
    raise RuntimeError("No path to objective")

class DisjointSet:
    def __init__(self):
        self.parent = {}

    def create_set(self, x):
        self.parent[x] = (x, 1)

    def find_set(self, x):
        node = x
        while self.parent[node][0] != node:
            node = self.parent[node][0]
        while self.parent[x][0] != x:
            x_next = self.parent[x][0]
            self.parent[x] = self.parent[node]
            x = x_next
        return node

    def union(self, x, y):
        x_root = self.find_set(x)
        y_root = self.find_set(y)
        x_height = self.parent[x_root][1]
        y_height = self.parent[y_root][1]
        if x_height > y_height:
            self.parent[y_root] = self.parent[x_root]
            self.parent[y] = self.parent[x_root]
        elif x_height < y_height:
            self.parent[x_root] = self.parent[y_root]
            self.parent[x] = self.parent[y_root]
        else:
            self.parent[y_root] = (x_root, x_height)
            self.parent[x_root] = (x_root, x_height + 1)

def shortest_path_all_dots(maze: maze.Maze):
    """Take a maze and return the shortest path between all pairs in start and objectives,
    in a form of an edge dictionary (key = (point, point), value = distance).
    """
    points = maze.getObjectives()
    point_set = set(points)
    edge_dict = {}
    for point in points:
        visited = set()
        queue = [(point, 0)]
        visited.add(point)
        num_objective_visited = 1
        while queue:
            (position, cost) = queue.pop(0)
            visited.add(position)
            if position in point_set:
                edge_dict[(point, position)] = cost
                edge_dict[(position, point)] = cost
                num_objective_visited += 1
            if num_objective_visited == len(points):
                break
            for neighbour in maze.getNeighbors(position[0], position[1]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, cost + 1))
    return edge_dict

Point = Tuple[int, int]

class Graph:
    vertices: List[Point]
    edge_weights: Dict[Tuple[Point, Point], int]
    def __init__(self, vertices, edge_weights):
        self.vertices = vertices
        self.edge_weights = edge_weights

def create_objective_graph(edge_dictionary, remaining_objectives):
    """Given the edge dictionary and the list of unvisited objectives, construct a graph useful for MST."""
    filtered_edge_dictionary = copy.copy(edge_dictionary)
    remaining_objectives = set(remaining_objectives)
    for (key, _) in edge_dictionary.items():
        if key[0] not in remaining_objectives or key[1] not in remaining_objectives:
            del filtered_edge_dictionary[key]
    return Graph(vertices=remaining_objectives, edge_weights=filtered_edge_dictionary)

def minimum_spanning_tree(objective_graph: Graph):
    """Return the cost of the MST of the graph."""
    sorted_edges: List[Tuple[Tuple[Point, Point], int]] = []
    for (key, value) in objective_graph.edge_weights.items():
        sorted_edges.append((key, value))
    sorted_edges.sort(key=lambda kv: kv[1])
    ds = DisjointSet()
    for v in objective_graph.vertices:
        ds.create_set(v)
    mst_cost = 0
    for (edge, cost) in sorted_edges:
        p1, p2 = edge
        if ds.find_set(p1) == ds.find_set(p2):
            continue
        mst_cost += cost
        ds.union(p1, p2)
    return mst_cost

def heuristic_part2(maze: maze.Maze, position, remaining_objectives, edge_dictionary):
    # compute MST on the graph
    mst_cost = minimum_spanning_tree(create_objective_graph(edge_dictionary, remaining_objectives))
    # compute shortest distance to any of the objectives
    dist = bfs_distance(maze, position, remaining_objectives)
    # heuristic cost is their sum
    return dist + mst_cost

def heuristic_extra(maze: maze.Maze, position, remaining_objectives, edge_dictionary):
    return bfs_distance(maze, position, remaining_objectives) + len(remaining_objectives) * 30
