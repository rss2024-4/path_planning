import numpy as np

class RRT:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.parent = None

    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        x_bounds,
        y_bounds,
        max_extend_length=3.0,
        path_resolution=0.5,
        goal_sample_rate=0.05,
        max_iter=1000,
    ):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.max_extend_length = max_extend_length
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.node_list = [self.start]
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(self.node_list, rnd_node)
            new = self.steer(nearest_node, rnd_node, max_extend_length=self.max_extend_length)
            if not self.collision(nearest_node, new, self.obstacle_list):
                self.node_list.append(new)

            # If the new_node is very close to the goal, connect it directly to the goal and return the final path
            if self.dist_to_goal(self.node_list[-1].p) <= self.max_extend_length:
                final_node = self.steer(self.node_list[-1], self.goal, self.max_extend_length)
                if not self.collision(final_node, self.node_list[-1], self.obstacle_list):
                    return self.final_path(len(self.node_list) - 1)
        return None  # cannot find path

    def steer(self, from_node, to_node, max_extend_length=np.inf):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length
        """
        new_node = self.Node(to_node.p)
        d = from_node.p - to_node.p
        dist = np.linalg.norm(d)
        if dist > max_extend_length:
            # rescale the path to the maximum extend_length
            new_node.p = from_node.p - d / dist * max_extend_length
        new_node.parent = from_node
        return new_node

    def dist_to_goal(self, p):
        """Distance from p to goal"""
        return np.linalg.norm(p - self.goal.p)

    def get_random_node(self):
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            # Sample random point inside boundaries
            rnd = self.Node([
                np.random.rand() * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0],
                np.random.rand() * (self.y_bounds[1] - self.y_bounds[0]) + self.y_bounds[0],
            ])
        else:
            # Select goal point
            rnd = self.Node(self.goal.p)
        return rnd

    @staticmethod
    def get_nearest_node(node_list, node):
        """Find the nearest node in node_list to node"""
        dlist = [np.sum(np.square((node.p - n.p))) for n in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    @staticmethod
    def collision(node1, node2, obstacle_list):
        """Check whether the path connecting node1 and node2
        is in collision with anyting from the obstacle_list
        """
        p1 = node2.p
        p2 = node1.p
        for o in obstacle_list:
            center_circle = o[0:2]
            radius = o[2]
            d12 = p2 - p1  # the directional vector from p1 to p2
            # defines the line v(t) := p1 + d12*t going through p1=v(0) and p2=v(1)
            d1c = center_circle - p1  # the directional vector from circle to p1
            # t is where the line v(t) and the circle are closest
            # Do not divide by zero if node1.p and node2.p are the same.
            # In that case this will still check for collisions with circles
            t = d12.dot(d1c) / (d12.dot(d12) + 1e-7)
            t = max(0, min(t, 1))  # Our line segment is bounded 0<=t<=1
            d = p1 + d12 * t  # The point where the line segment and circle are closest
            is_collide = np.sum(np.square(center_circle - d)) < radius**2
            if is_collide:
                return True  # is in collision
        return False  # is not in collision

    def final_path(self, goal_ind):
        """Compute the final path from the goal node to the start node"""
        path = [self.goal.p]
        node = self.node_list[goal_ind]
        while node != self.start:
            path.append(node.p)
            node = node.parent
        path.append(node.p)
        return path

    # def draw_graph(self):
    #     for node in self.node_list:
    #         if node.parent:
    #             plt.plot(
    #                 [node.p[0], node.parent.p[0]],
    #                 [node.p[1], node.parent.p[1]],
    #                 "-g",
    #             )

class RRTStar(RRT):
    class Node(RRT.Node):
        def __init__(self, p):
            super().__init__(p)
            self.cost = 0.0

    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        x_bounds,
        y_bounds,
        max_extend_length=5.0,
        path_resolution=0.5,
        goal_sample_rate=0.0,
        max_iter=1000,
        connect_circle_dist=50.0,
    ):
        super().__init__(
            start,
            goal,
            obstacle_list,
            x_bounds,
            y_bounds,
            max_extend_length,
            path_resolution,
            goal_sample_rate,
            max_iter,
        )
        self.connect_circle_dist = connect_circle_dist
        self.goal = self.Node(goal)

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # Create a random node inside the bounded environment
            rnd = self.get_random_node()
            # Find nearest node
            nearest_node = self.get_nearest_node(self.node_list, rnd)
            # Get new node by connecting rnd_node and nearest_node
            new_node = self.steer(nearest_node, rnd, self.max_extend_length)
            # If path between new_node and nearest node is not in collision:
            if not self.collision(new_node, nearest_node, self.obstacle_list):
                near_inds = self.near_nodes_inds(new_node)
                # Connect the new node to the best parent in near_inds
                new_node = self.choose_parent(new_node, near_inds)
                self.node_list.append(new_node)
                # Rewire the nodes in the proximity of new_node if it improves their costs
                self.rewire(new_node, near_inds)
        last_index, min_cost = self.best_goal_node_index()
        if last_index:
            return self.final_path(last_index), min_cost
        return None, min_cost

    def choose_parent(self, new_node, near_inds):
        """Set new_node.parent to the lowest resulting cost parent in near_inds and
        new_node.cost to the corresponding minimal cost
        """
        min_cost = np.inf
        best_near_node = None
        for i in near_inds:
            node = self.node_list[i]
            if not self.collision(new_node, node, self.obstacle_list):
                new_cost = self.new_cost(node, new_node)
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_near_node = node
                
        new_node.cost = min_cost
        new_node.parent = best_near_node
        return new_node

    def rewire(self, new_node, near_inds):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        for i in near_inds:
            node = self.node_list[i]
            new_cost = self.new_cost(new_node, node)
            if not self.collision(node, new_node, self.obstacle_list) and new_cost < node.cost:
                node.cost = new_cost
                node.parent = new_node                

        self.propagate_cost_to_leaves(new_node)

    def best_goal_node_index(self):
        """Find the lowest cost node to the goal"""
        min_cost = np.inf
        best_goal_node_idx = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            # Has to be in close proximity to the goal
            if self.dist_to_goal(node.p) <= self.max_extend_length:
                # Connection between node and goal needs to be collision free
                if not self.collision(self.goal, node, self.obstacle_list):
                    # The final path length
                    cost = node.cost + self.dist_to_goal(node.p)
                    if node.cost + self.dist_to_goal(node.p) < min_cost:
                        # Found better goal node
                        min_cost = cost
                        best_goal_node_idx = i
        return best_goal_node_idx, min_cost

    def near_nodes_inds(self, new_node):
        """Find the nodes in close proximity to new_node"""
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * np.sqrt((np.log(nnode) / nnode))
        dlist = [np.sum(np.square((node.p - new_node.p))) for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r**2]
        return near_inds

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        d = np.linalg.norm(from_node.p - to_node.p)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)