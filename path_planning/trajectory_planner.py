import rclpy
from rclpy.node import Node
import numpy as np

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
from .rrt import RRT


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

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.x_bounds = (-50, 50)
        self.y_bounds = (-50, 50)
        self.obstacles = [] #x,y,radius
        self.start = None

    def map_cb(self, msg):
        raise NotImplementedError

    def pose_cb(self, pose):
        self.start = [pose.pose.pose.position.x, pose.pose.pose.position.y]

    def goal_cb(self, msg):
        goal = [msg.pose.position.x, msg.pose.position.y]
        rrt = RRT(self.start, goal, self.obstacles, self.x_bounds, self.y_bounds)
        self.trajectory.points = rrt.plan()
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()