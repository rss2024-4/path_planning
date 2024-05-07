import numpy as np
import heapq
import math
import os
import json

class ASTAR:

    def __init__(self, obstacles, path, logger):
        self.centerline = []
        self.load_centerline(path)

        # self.grid = self.Grid(obstacles, self.centerline, logger, .125)
        self.grid = self.Grid(obstacles, self.centerline, logger, 0.125)

        self.logger = logger

    def load_centerline(self, path):
        # path = "/home/racecar/racecar_ws/install/path_planning/share/path_planning/example_trajectories/full_lane.traj"
        # print("Loading trajectory:", path)

        # resolve all env variables in path
        path = os.path.expandvars(path)

        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                self.centerline.append(np.array((p["x"], p["y"])))
    class Node:
        def __init__(self, x, y, obstacle=False):
            self.x = x
            self.y = y
            self.obstacle = obstacle
            self.g = float('inf')  # distance from start node
            self.h = float('inf')  # heuristic distance to goal node
            self.parent = None
            self.direction = None

        def __lt__(self, other):
            return (self.g + self.h) < (other.g + other.h)

    class Grid:
        def __init__(self, obstacles, centerline, logger, cell_size=1):
            #  list of points representing the centerline
            self.centerline = centerline
            self.cell_size = cell_size
            self.width_min = -62.0
            self.width_max = 27.0
            self.height_min = -5.0
            self.height_max = 40.0
            # self.width_min = -60.0
            # self.width_max = 0.
            # self.height_min = -4.0
            # self.height_max = 40.0 
            self.logger = logger
            width_range = np.linspace(self.width_min, self.width_max, (self.width_max - self.width_min)/self.cell_size + 1)
            height_range = np.linspace(self.height_min, self.height_max, (self.height_max - self.height_min)/self.cell_size + 1)
            self.nodes = {(x, y): ASTAR.Node(x, y) for x in width_range for y in height_range}
            self.place_obstacles(obstacles)
            self.calculate_vectors()

        
        # for vector field processing
        def dist2(self, p1, p2):
            return (p1[0]-p2[0])**2 + (p1[1] - p2[1])**2
    
        def lineSegToPoint2(self, start, end, p):
            l = self.dist2(start, end)
            t = max(0, min(1, np.dot(p-start, end-start)/l))
            projection = start + t * (end-start)
            return self.dist2(p, projection), projection

        def calculate_vectors(self):
            # print('calculating vectors')
            self.logger.info("calculating vectors")
            for p in self.nodes:
                if self.nodes[p].obstacle:
                    continue
                minDist = None
                # closestPoint = None
                closestIdx = None

                for i in range(len(self.centerline)-1):
                    start, end = self.centerline[i], self.centerline[i+1]
                    dist, projection = self.lineSegToPoint2(start, end, p)
                    if not minDist or dist < minDist:
                        minDist = dist
                        # closestPoint = projection
                        closestIdx = i
                    
                centerline_vec = np.array((self.centerline[closestIdx][0] - self.centerline[closestIdx+1][0], \
                                  self.centerline[closestIdx][1] - self.centerline[closestIdx+1][1]))
                grid_vec = np.array((p[0] - self.centerline[closestIdx][0], \
                            p[1] - self.centerline[closestIdx][1]))
                cross = np.cross(grid_vec, centerline_vec)
                # self.logger.info(f'cross: {cross}')
                if cross > 0:
                    self.nodes[p].direction = centerline_vec / np.linalg.norm(centerline_vec)
                else:
                    self.nodes[p].direction = -1*centerline_vec / np.linalg.norm(centerline_vec)

            self.logger.info("done calculating vectors")
        def place_obstacles(self, obstacles):
            for obstacle in obstacles:
                x = obstacle[0]
                y = obstacle[1]
                # rad = obstacle[2]
                rad = .1
                x_range = (x-rad, x+rad)
                y_range = (y-rad, y+rad)
                x_min = x-rad
                x_max = x+rad
                y_min = y-rad
                y_max = y+rad
                cells_per_meter = 1/self.cell_size
                x_min_rounded = round(x_min * cells_per_meter) / cells_per_meter
                x_max_rounded = round(x_max * cells_per_meter) / cells_per_meter
                y_min_rounded = round(y_min * cells_per_meter) / cells_per_meter
                y_max_rounded = round(y_max * cells_per_meter) / cells_per_meter
                x_locs = np.linspace(x_min_rounded, x_max_rounded, int((x_max_rounded - x_min_rounded)*cells_per_meter + 1))
                y_locs = np.linspace(y_min_rounded, y_max_rounded, int((y_max_rounded - y_min_rounded)*cells_per_meter + 1))

                for x in x_locs:
                    for y in y_locs:
                        loc = (x,y)
                        self.nodes[loc].obstacle = True
                
                # for loc in self.nodes:
                #     node_x = loc[0]
                #     node_y = loc[1]
                #     if x_range[0] < node_x and x_range[1] > node_x and y_range[0] < node_y and y_range[1] > node_y:
                #         self.nodes[(node_x, node_y)].obstacle = True
            return
        
        def __getitem__(self, coordinates):
            x, y = coordinates
            return self.nodes[(x, y)]
        
        def is_close_to(self, x, y, epsilon = 0.0001):
            return x > y-epsilon and x < y + epsilon
            

        def get_neighbors(self, node):
            # returns neighbors and cost of moving to neighbor
            neighbors = []
            directions = [(self.cell_size, 0), (0, self.cell_size), (-self.cell_size, 0), (0, -self.cell_size)]  # right, down, left, up
            directions_valid = [False,False,False,False]
            for i in range(len(directions)):
                dx, dy = directions[i]
                new_x, new_y = node.x + dx, node.y + dy
                if (self.width_min <= new_x < self.width_max and self.height_min <= new_y < self.height_max) and not self.nodes[(new_x, new_y)].obstacle:
                    if (not self.opposite_dir(node, directions[i])):
                        neighbors.append((self.nodes[(new_x, new_y)],self.cell_size))
                        directions_valid[i] = True

            # account for diagonals!! 
            prev_dir_valid = directions_valid[3]
            prev_dir = directions[3]
            for i in range(len(directions)):
                cur_dir_valid = directions_valid[i]
                cur_dir = directions[i]
                if cur_dir_valid and prev_dir_valid:
                    diagonal_dx = prev_dir[0] + cur_dir[0]
                    diagonal_dy = prev_dir[1] + cur_dir[1]
                    new_x, new_y = node.x + diagonal_dx, node.y + diagonal_dy
                    if (self.width_min <= new_x < self.width_max and \
                        self.height_min <= new_y < self.height_max) and \
                        not self.nodes[(new_x, new_y)].obstacle and \
                        not (self.is_close_to(node.direction[0],-1 * self.nodes[(new_x, new_y)].direction[0]) and \
                             self.is_close_to(node.direction[1], -1 *self.nodes[(new_x, new_y)].direction[1])):
                        neighbors.append((self.nodes[(new_x, new_y)], np.sqrt(2)*self.cell_size))
                prev_dir_valid = cur_dir_valid
                prev_dir = cur_dir
            return neighbors
        
        def get_node_from_loc(self, loc):
            x = loc[0]
            y = loc[1]
            half_cell = self.cell_size/2 
            for grid_loc in self.nodes:
                grid_x = grid_loc[0]
                grid_y = grid_loc[1]
                if x > grid_x - half_cell and x < grid_x + half_cell and y > grid_y - half_cell and y < grid_y + half_cell:
                    return self.nodes[grid_loc]
                
        def obstacles_along_path(self, pt1, pt2):
            cells_per_meter = 1/self.cell_size
            dis = math.floor(math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))
            x_locs = np.linspace(pt1[0], pt2[0], 20*dis)
            y_locs = np.linspace(pt1[1], pt2[1], 20*dis)
            n1 = self.nodes[pt1]
            for i in range(20*dis):
                x_rounded = round(x_locs[i] * cells_per_meter) / cells_per_meter
                y_rounded = round(y_locs[i] * cells_per_meter) / cells_per_meter
                loc = (x_rounded, y_rounded)
                n2 = self.nodes[loc].obstacle
                direction = (n2[0]-n1[0], n2[1]-n1[1])
                if n2.obstacle == True and not self.opposite_dir(n1, direction):
                    return True
            return False


    def heuristic(self, node, goal):
        return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)

    def reconstruct_path(self, current):
        path = []
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]

    
    def optimize_path(self, path):
        new_path = []
        i = 0
        while True:
            pt1 = path[i]
            new_path.append(pt1)
            if i == len(path)-1:
                return new_path
            shortcut_node = None
            shortcut_j = None
            j = i+1
            # find a pt2 where pt1 to pt2 is obstacle free
            while True:
                pt2 = path[j]
                shortcut = not self.grid.obstacles_along_path(pt1, pt2)
                if shortcut:
                    shortcut_node = pt2
                    shortcut_j = j
                j += 1
                if j >= len(path):
                    break
            if shortcut_node:
                i = shortcut_j
            else:
                i += 1
    

    def plan(self, start_loc, goal_loc):
        open_set = []
        closed_set = set()

        start = self.grid.get_node_from_loc(start_loc)
        goal = self.grid.get_node_from_loc(goal_loc)

        start.g = 0
        start.h = self.heuristic(start, goal)
        heapq.heappush(open_set, start)

        while open_set:
            current = heapq.heappop(open_set)

            if current == goal:
                path = self.reconstruct_path(current)
                return self.optimize_path(path)

            closed_set.add(current)

            for neighbor, neighbor_g in self.grid.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = current.g + neighbor_g

                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)

        return None  # No path found

# # Example usage:
# grid = Grid(width=3, height=3, cell_size=1)
# # Add obstacles if needed, e.g., grid[(1, 1)].obstacle = True

# start_node = grid[0, 0]
# goal_node = grid[2, 2]

# path = astar(start_node, goal_node, grid)
# if path:
#     print("Path found:", path)
# else:
#    print("No path found")
