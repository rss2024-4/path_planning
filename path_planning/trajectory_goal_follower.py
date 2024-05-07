import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, Point, PointStamped, PoseArray, Pose
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from tf_transformations import euler_from_quaternion

import numpy as np

from .astar import ASTAR

from .utils import LineTrajectory
from visualization_msgs.msg import Marker

import tf_transformations as tf

DEFAULT = 0
WAIT_FOR_PLAN = 1
PLANNED_PATH = 2

class PurePursuitWithTargets(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_goal_follower")
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('drive_topic', "/drive")
        self.declare_parameter('centerline', 'default')

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.centerline = self.get_parameter('centerline').get_parameter_value().string_value
 
        self.get_logger().info(f'{self.drive_topic}, {self.odom_topic}')

        self.lookahead = 0.8  # FILL IN #
        self.speed = 1  # FILL IN #
        self.wheelbase_length = 0.35 # FILL IN #
        self.default_points = []
        self.default_visited = []
        self.initialized_traj = False

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        
        self.goal_traj_sub = self.create_subscription(PoseArray,
                                                 "/goal_trajectory/current",
                                                 self.goal_trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.odom_callback,
                                                 1)
        
        self.point_pub = self.create_publisher(Marker, "lookahead", 1)


        # goal stuff
        self.goalpoint_sub = self.create_subscription(PointStamped,
                                                    "/clicked_point",
                                                    self.goal_cb,
                                                    1)
        
        # publishes start and goal points for the astar planner
        self.planner_pub = self.create_publisher(PoseArray, "/plan", 1)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_cb,
            1
        )

        self.goal_points = []
        self.state = DEFAULT
        self.initialized_map = False
        self.goal_trajectory = LineTrajectory(node=self, viz_namespace="/goal_trajectory")

    def map_cb(self, msg):
        self.get_logger().info("Processing Map")
        T = self.pose_to_T(msg.info.origin)
        table = np.array(msg.data)
        table = table.reshape((msg.info.height, msg.info.width))
        obstacles = []
        for i, row in enumerate(table):
            for j, val in enumerate(row):
                if val > 85:
                    px = np.eye(3)
                    px[1,2] = i*msg.info.resolution
                    px[0,2] = j*msg.info.resolution
                    p = T@px
                    obstacles.append([p[0,2], p[1,2], 0.25])

        # for testing vectors
        self.astar = ASTAR(obstacles, self.centerline, self.get_logger()) 
        self.initialized_map = True
        self.get_logger().info("Map processed")

    def goal_cb(self, msg):
        self.get_logger().info(f'got goal point: {msg.point}')
        # point and whether or not it was already visited
        self.goal_points.append(((msg.point.x, msg.point.y), False))

    def dist2(self, p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1] - p2[1])**2

    def odom_callback(self, msg):
        # self.get_logger().info("received odom msg")
        xpos = msg.pose.pose.position.x
        ypos = msg.pose.pose.position.y
        theta = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])[2]
        p = np.array([xpos, ypos])

        if not self.initialized_traj or not self.initialized_map:
            # self.get_logger().info("no trajectory info")
            return
        
        # self.get_logger().info(str(self.visited))
        points = self.default_points
        visited = self.default_visited

        if self.state == DEFAULT:
            minDist = float('inf')
            goal = None
            for gp, seen in self.goal_points:
                if seen:
                    continue

                d = self.dist2(p, gp)
                if d < 12 and d < minDist:
                    minDist = d
                    goal = gp
            
            if goal is not None:
                self.get_logger().info(f'close to goal point: {goal}')
                # stop
                drive_cmd = AckermannDriveStamped()
                drive_cmd.drive.speed = 0.0
                self.drive_pub.publish(drive_cmd)

                traj = self.astar.plan(p, gp)
                if traj is not None:
                    self.state = PLANNED_PATH
                    self.get_logger().info(f"Receiving new trajectory to goal pose {gp} points")

                    self.goal_trajectory.clear()
                    # self.goal_trajectory.fromPoseArray(msg)
                    self.goal_trajectory.points = traj
                    self.goal_trajectory.publish_viz()

                    self.goal_points = np.array(self.goal_trajectory.points)
                    self.goal_visited = np.array([False] * len(self.goal_trajectory.points))
        elif self.state == PLANNED_PATH:
            points = self.goal_points
            visited = self.goal_visited

        drive_cmd = AckermannDriveStamped()
        
        target = None
        for i in range(len(points)):
            if not visited[i]:
                if self.dist2(points[i], p) < self.lookahead ** 2:
                    visited[i] = True
                else:
                    target = points[i]
                    break
        
        if target is None:
            # self.get_logger().info("no target")
            drive_cmd.drive.speed = 0.0
        else:
            
            angle = self.find_steering_angle(p, theta, target)
            if np.isnan(angle):
                # self.get_logger().info("null angle")
                angle = 0.0
            if np.abs(angle) < 0.05:
                drive_cmd.drive.speed = 2.0
            else:
                drive_cmd.drive.speed = self.speed*1.0

            # self.get_logger().info(f'angle: {angle}, speed: {drive_cmd.drive.speed}, target: {target}')
            drive_cmd.drive.steering_angle = angle
            self.publish_point(target)
            
        self.drive_pub.publish(drive_cmd)

    def find_steering_angle(self, p, theta, lookaheadPoint):
        target = lookaheadPoint - p
        car_vec = np.cos(theta), np.sin(theta) # vector of car
        
        # steer
        d = np.linalg.norm(target)
        eta = np.arccos(np.dot(car_vec, target)/(d * np.linalg.norm(car_vec)))
        delta = np.arctan(2*self.wheelbase_length*np.sin(eta))
        sign = np.sign(np.cross(car_vec, target))
        return delta*sign

    def publish_point(self, p):
        # Construct a line
        msg = Marker()
        msg.type = Marker.POINTS
        msg.header.frame_id = "map"

        # Set the size and color
        msg.scale.x = 0.1
        msg.scale.y = 0.1
        msg.color.a = 1.
        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.g = 1.0

        # Fill the line with the desired values
        pt = Point()
        pt.x = p[0]
        pt.y = p[1]
        msg.points.append(pt)

        # Publish the line
        self.point_pub.publish(msg)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz()

        self.default_points = np.array(self.trajectory.points)
        self.default_visited = np.array([False] * len(self.trajectory.points))

        self.initialized_traj = True
    
    def goal_trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory to goal pose {len(msg.poses)} points")

        self.goal_trajectory.clear()
        self.goal_trajectory.fromPoseArray(msg)
        self.goal_trajectory.publish_viz()

        self.goal_points = np.array(self.goal_trajectory.points)
        self.goal_visited = np.array([False] * len(self.goal_trajectory.points))

        self.state = PLANNED_PATH
    
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


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuitWithTargets()
    rclpy.spin(follower)
    rclpy.shutdown()
