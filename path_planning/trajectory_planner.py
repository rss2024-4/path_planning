import rclpy
from rclpy.node import Node
import numpy as np
import yaml

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, Point
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
from .rrt import RRT, RRTStar
from .astar import ASTAR
import tf_transformations as tf
from std_msgs.msg import Header


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('centerline', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.centerline = self.get_parameter('centerline').get_parameter_value().string_value

        self.method = "astar"
        # self.method = "rrt"

        self.get_logger().info(f'Path center: {self.centerline}')

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_cb,
            10
        )

        self.obstacles_pub = self.create_publisher(
            MarkerArray,
            "/obstacles",
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.x_bounds = (-61, 25) # in meters
        self.y_bounds = (-16, 48) # in meters
        self.obstacles = [] #x,y,radius (in meters)
        self.start = None
        self.obstacle_radius = .25

        self.create_timer(0.05, self.timer_cb)

    def publish_vectors(self, vectors):
        id = 3
        markers = []
        for start, end in vectors:
            marker = Marker()
            stamp = self.get_clock().now().to_msg()
            header = Header()
            header.stamp = stamp
            header.frame_id = "map"

            marker.header = header
            marker.id = id
            marker.type = marker.ARROW  # line strip

            start_point = Point()
            start_point.x = start[0]
            start_point.y = start[1]

            end_p = Point()
            end_p.x = end[0]
            end_p.y = end[1]
            
            marker.points = [start_point, end_p]
            marker.scale.x = 0.1
            marker.scale.y = 0.3
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            id += 1

            markers.append(marker)

        markers_msg = MarkerArray()
        markers_msg.markers = markers
        self.vec_pub.publish(markers_msg)

    def map_cb(self, msg):
        self.get_logger().info("Processing Map")
        T = self.pose_to_T(msg.info.origin)
        table = np.array(msg.data)
        table = table.reshape((msg.info.height, msg.info.width))
        # if self.method == "rrt":
        for i, row in enumerate(table):
            for j, val in enumerate(row):
                if val > 85:
                    px = np.eye(3)
                    px[1,2] = i*msg.info.resolution
                    px[0,2] = j*msg.info.resolution
                    p = T@px
                    x, y = p[0,2], p[1,2]
                    self.obstacles.append([float(x), float(y), self.obstacle_radius])
                    # if len(self.obstacles) == 0:
                    #     self.obstacles.append([float(x), float(y), self.obstacle_radius])
                    # else:
                    #     for i in range(len(self.obstacles)):
                    #         if not self.collision(x, y, self.obstacles[i]):
                    #             self.obstacles.append([float(x), float(y), self.obstacle_radius])
                    #             break
        with open("src/path_planning/maps/obstacles.yaml", 'w') as file:
            yaml.dump(self.obstacles, file, default_flow_style=False)
        self.get_logger().info("Map processed")

    def collision(self, x, y, obs):
        if np.sqrt((x-obs[0])**2 + (y-obs[1])**2) <= 2*self.obstacle_radius:
            return True
        return False

    def pose_cb(self, pose):
        self.start = [pose.pose.pose.position.x, pose.pose.pose.position.y]
        

    def goal_cb(self, msg):
        goal = [msg.pose.position.x, msg.pose.position.y]

        if self.method == "astar":
            self.get_logger().info("Start: (%s,%s)" % (self.start[0],self.start[1]))
            self.get_logger().info("Goal: (%s,%s)" % (goal[0],goal[1]))
            astar = ASTAR(self.obstacles, self.start, goal, self.get_logger())   
            # for p in astar.nodes:
            #     start = p
            #     end = np.array(p) + p.direction
            #     self.publish_vector(start, end) 

            # self.get_logger().info(",".join(str(loc)+str(astar.grid.nodes[loc].obstacle) for loc in astar.grid.nodes))
            self.get_logger().info("Finding path")
            traj = astar.plan(self.start, goal)
            if not traj:
                self.get_logger().info("No Path Found.")
                return
            self.trajectory.points = traj
            # self.get_logger().info(str(traj))
            self.get_logger().info(f"Path found: {traj}")
            
            self.traj_pub.publish(self.trajectory.toPoseArray())
            # self.trajectory.publish_viz()
        
        elif self.method == "rrt":
            rrt = RRT(self.start, goal, self.obstacles, self.x_bounds, self.y_bounds)
            # rrt = RRTStar(self.start, goal, self.obstacles, self.x_bounds, self.y_bounds)
            
            self.get_logger().info("Finding path")
            traj = rrt.plan()
            if not traj:
                self.get_logger().info("No Path Found.")
                return
            self.trajectory.points = traj
            self.get_logger().info(f"Path found: {traj}")
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()

    def pose_to_T(self, pose_msg):
        th = tf.euler_from_quaternion([
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ])[2]
        x, y = pose_msg.position.x, pose_msg.position.y
        return np.array([
            [np.cos(th), -np.sin(th), x],
            [np.sin(th),  np.cos(th), y],
            [         0,           0, 1],
        ])
    
    def arr_to_pose_arr(self, arr):
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        a = []
        for i in arr:
            pose = Pose()
            pose.position.x = i[0]
            pose.position.y = i[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            a.append(pose)
        msg.poses = a
        return msg
    
    def arr_to_marker_arr(self, arr):
        msg = MarkerArray()
        a = []
        for i, obs in enumerate(arr):
            marker = Marker()
            marker.type = 3
            marker.id = i
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.pose.position.x = obs[0]
            marker.pose.position.y = obs[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = obs[2]
            marker.scale.y = obs[2]
            marker.scale.z = obs[2]
            marker.color.a = 1.
            marker.color.r = 255.
            marker.color.g = 0.
            marker.color.b = 0.
            a.append(marker)
        msg.markers = a
        return msg

    def timer_cb(self):
        if len(self.obstacles) == 0:
            return
        self.obstacles_pub.publish(self.arr_to_marker_arr(self.obstacles))


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()