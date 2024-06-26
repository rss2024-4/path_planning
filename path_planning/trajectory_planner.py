import rclpy
from rclpy.node import Node
import numpy as np

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
from .rrt import RRT, RRTStar
from .astar import ASTAR
import tf_transformations as tf


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.method = "astar"
        # self.method = "rrt"

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
            PoseArray,
            "/obstacles",
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.x_bounds = (-61, 25) # in meters
        self.y_bounds = (-16, 48) # in meters
        self.obstacles = [] #x,y,radius (in meters)
        self.start = None

        # self.create_timer(0.05, self.timer_cb)

    def map_cb(self, msg):
        self.get_logger().info("Processing Map")
        T = self.pose_to_T(msg.info.origin)
        table = np.array(msg.data)
        table = table.reshape((msg.info.height, msg.info.width))
        for i, row in enumerate(table):
            for j, val in enumerate(row):
                if val > 85:
                    px = np.eye(3)
                    px[1,2] = i*msg.info.resolution
                    px[0,2] = j*msg.info.resolution
                    p = T@px
                    self.obstacles.append([p[0,2], p[1,2], 0.25])
        # self.get_logger().info(str(min([element[0] for element in self.obstacles])))
        # self.get_logger().info(str(max([element[0] for element in self.obstacles])))
        # self.get_logger().info(str(min([element[1] for element in self.obstacles])))
        # self.get_logger().info(str(max([element[1] for element in self.obstacles])))
        # self.get_logger().info(",".join(str(element) for element in self.obstacles))
        self.get_logger().info("Map processed")

    def pose_cb(self, pose):
        self.start = [pose.pose.pose.position.x, pose.pose.pose.position.y]
        

    def goal_cb(self, msg):

        self.method = "astar"

        goal = [msg.pose.position.x, msg.pose.position.y]

        if self.method == "astar":
            self.get_logger().info("Start: (%s,%s)" % (self.start[0],self.start[1]))
            self.get_logger().info("Goal: (%s,%s)" % (goal[0],goal[1]))
            astar = ASTAR(self.obstacles, self.start, goal)        
            # self.get_logger().info(",".join(str(loc)+str(astar.grid.nodes[loc].obstacle) for loc in astar.grid.nodes))
            self.get_logger().info("Finding path")
            traj = astar.plan(self.start, goal)
            self.trajectory.points = traj
            # self.get_logger().info(str(traj))
            self.get_logger().info(f"Path found: {traj}")
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        
        elif self.method == "rrt":
            rrt = RRT(self.start, goal, self.obstacles, self.x_bounds, self.y_bounds)
            rrt = RRTStar(self.start, goal, self.obstacles, self.x_bounds, self.y_bounds)
            
            self.get_logger().info("Finding path")
            traj = rrt.plan()
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

    def timer_cb(self):
        if len(self.obstacles) == 0:
            return
        self.obstacles_pub.publish(self.arr_to_pose_arr(self.obstacles))


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()