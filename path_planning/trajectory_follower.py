import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

import numpy as np

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    END_DIST = 0.05

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value


        self.lookahead = 0.8  # FILL IN #
        self.speed = 1  # FILL IN #
        self.wheelbase_length = 0.35 # FILL IN #
        self.points = []
        self.visited = []
        self.initialized_traj = False

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.odom_callback,
                                                 1)
        

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

        if not self.initialized_traj:
            # self.get_logger().info("no trajectory info")
            return
        
        self.get_logger().info(str(self.visited))

        target = None
        
        for i in range(len(self.points)):
            if not self.visited[i]:
                if self.dist2(self.points[i], p) < self.lookahead ** 2:
                    self.visited[i] = True
                else:
                    target = self.points[i]
                    break
                
        if all(self.visited[i]) and self.dist2(self.points[-1], p) > self.END_DIST:
            target = self.points[-1]
                
        drive_cmd = AckermannDriveStamped()
        
        if target is None:
            drive_cmd.drive.speed = 0.0
        else:
            angle = self.find_steering_angle(p, theta, target)
            drive_cmd.drive.steering_angle = angle
            drive_cmd.drive.speed = self.speed*1.0
            
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
        

    def pose_callback_fast(self, msg):
        pass


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz()

        self.points = np.array(self.trajectory.points)
        self.visited = np.array([False] * len(self.trajectory.points))
        # self.get_logger().info(f'Points: {",".join(self.trajectory.points)}')
        for p in self.points:
            self.get_logger().info(str(p))

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
