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
import os
import json

DEFAULT = 0
WAIT_FOR_PLAN = 1
PLANNED_PATH = 2

RIGHT = 'RIGHT'
LEFT = 'LEFT'

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
        self.centerline_path = self.get_parameter('centerline').get_parameter_value().string_value
        self.centerline = []

        self.load_centerline(self.centerline_path)
 
        self.get_logger().info(f'{self.drive_topic}, {self.odom_topic}')

        self.lookahead = 0.8  # FILL IN #
        self.speed = 1  # FILL IN #
        self.wheelbase_length = 0.35 # FILL IN #
        self.default_points = []
        self.default_visited = []
        self.initialized_traj = False

        # which goal point to follow currently
        self.goal_idx = 0

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
        
        self.point_pub = self.create_publisher(Marker, "/lookahead", 1)


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

        self.start_wait = None

    def load_centerline(self, path):
        path = os.path.expandvars(path)

        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                self.centerline.append(np.array((p["x"], p["y"])))

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
        self.astar = ASTAR(obstacles, self.centerline_path, self.get_logger()) 
        # self.astar = ASTAR(obstacles) 
        self.initialized_map = True
        self.get_logger().info("Map processed")
    
    def lineSegToPoint2(self, start, end, p):
            l = self.dist2(start, end)
            t = max(0, min(1, np.dot(p-start, end-start)/l))
            projection = start + t * (end-start)
            return self.dist2(p, projection), projection
    
    def centerline_side(self, p):
        self.get_logger().info(f'centerline side {p}')
        minDist = None
        closestIdx = None

        for i in range(len(self.centerline)-1):
            start, end = self.centerline[i], self.centerline[i+1]
            self.get_logger().info(f'centerline side {start}, {end}')
            dist, projection = self.lineSegToPoint2(start, end, p)
            if not minDist or dist < minDist:
                minDist = dist
                closestIdx = i
            
            # maybe this doesn't work if you're getting farther from each segment stop b/c u prob found closest
            if dist > minDist + 2:
                break
            
        centerline_vec = np.array((self.centerline[closestIdx][0] - self.centerline[closestIdx+1][0], \
                            self.centerline[closestIdx][1] - self.centerline[closestIdx+1][1]))
        grid_vec = np.array((p[0] - self.centerline[closestIdx][0], \
                    p[1] - self.centerline[closestIdx][1]))
        cross = np.cross(grid_vec, centerline_vec)

        if cross > 0:
            return RIGHT
        else:
            return LEFT

    def goal_cb(self, msg):
        self.get_logger().info(f'got goal point: {msg.point}')
        p = (msg.point.x, msg.point.y)
        self.goal_points.append(((msg.point.x, msg.point.y), self.centerline_side(p)))

    def dist2(self, p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1] - p2[1])**2

    def odom_callback(self, msg):
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
            # self.get_logger().info("no trajectory info, no map, or no goal points")
            return
        
        if len(self.goal_points) != 3:
            # self.get_logger().info("not enough goal points")
            return
        
        # self.get_logger().info(str(self.visited))
        points = self.default_points
        visited = self.default_visited

        if self.state == DEFAULT:
            # self.get_logger().info("following default")
            minDist = float('inf')
            goal = None


            # check if close to next goal point
            d = self.dist2(p, self.goal_points[self.goal_idx][0])
            if d < 12 and d < minDist:
                minDist = d
                goal, goal_side = self.goal_points[self.goal_idx]
                self.get_logger().info(f'goalidx, {self.goal_idx}')
            
            # if close, then check that car is the on the same side
            if goal is not None:
                self.get_logger().info(f'close to goal point: {goal}')

                car_side = self.centerline_side(p)

                if car_side == goal_side:
                    self.get_logger().info(f'car on same side as goal, go to it')
                    # stop
                    drive_cmd = AckermannDriveStamped()
                    drive_cmd.drive.speed = 0.0
                    self.drive_pub.publish(drive_cmd)

                    # plan a traj
                    traj = self.astar.plan(p, goal)
                    if traj is not None:
                        self.state = PLANNED_PATH
                        self.get_logger().info(f"Receiving new trajectory to goal pose {goal} points")

                        self.goal_trajectory.clear()
                        # self.goal_trajectory.fromPoseArray(msg)
                        self.goal_trajectory.points = traj
                        self.goal_trajectory.publish_viz()

                        self.goal_path = np.array(self.goal_trajectory.points)
                        self.goal_visited = np.array([False] * len(self.goal_trajectory.points))
                    else:
                        self.get_logger().info(f'could not get path to goal')
                else:
                    self.get_logger().info(f'car not on same side as goal')
        # if im going towards a goal point, follow those points instead
        elif self.state == PLANNED_PATH:
            points = self.goal_path
            visited = self.goal_visited

        drive_cmd = AckermannDriveStamped()
        
        # find target
        target = None
        for i in range(len(points)):
            if not visited[i]:
                if self.dist2(points[i], p) < self.lookahead ** 2:
                    visited[i] = True
                else:
                    target = points[i]
                    break
        
        # if this means im within lookahead distance if my goal point
        if self.state == PLANNED_PATH and target is None:
            if self.dist2(points[-1], p) < 0.15:
                # self.get_logger().info(f'reached goal point {p}')
                drive_cmd.drive.speed = 0.0
                self.drive_pub.publish(drive_cmd)

                if self.start_wait is None:
                    self.start_wait = self.get_clock().now().to_msg().sec
                while self.get_clock().now().to_msg().sec - self.start_wait < 3:
                    return
                
                # increase goal index
                if self.goal_idx < 2:
                    self.goal_idx += 1
                    self.get_logger().info(f'incrementing goal idx to {self.goal_idx}')
                    self.state = DEFAULT
                    self.start_wait = None
                return
            else:
                target = points[-1]
        
        # otherwise stop/ find the correct angle
        if target is None:
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
        msg.scale.x = 0.15
        msg.scale.y = 0.15
        msg.color.a = 1.
        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0

        # Fill the line with the desired values
        pt = Point()
        pt.x = p[0]
        pt.y = p[1]
        msg.points = [pt]
        # msg.points.append(pt)

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
