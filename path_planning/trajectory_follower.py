import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PointStamped
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

import numpy as np

from .utils import LineTrajectory, CubicHermiteGroup

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    END_DIST = 0.25

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value


        self.lookahead = 1.5  # FILL IN #
        self.speed = 1  # FILL IN #
        self.wheelbase_length = 0.325 # FILL IN #
        self.points = []
        self.spline = None
        self.initialized_traj = False

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.lookahead_pub = self.create_publisher(PointStamped, 'lookahead', 10)
        self.closest_pub = self.create_publisher(PointStamped, 'closest', 10)
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
        
        t = self.spline.getClosestPointT(p)
        closest = self.spline.get(t)
        target = self.spline.getLookaheadPointFromT(t, p, self.lookahead)
        
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'map'
        point.point.x = target[0]
        point.point.y = target[1]
        point.point.z = 0.0
        
        point2 = PointStamped()
        point2.header.stamp = self.get_clock().now().to_msg()
        point2.header.frame_id = 'map'
        point2.point.x = closest[0]
        point2.point.y = closest[1]
        point2.point.z = 0.0
        
        self.lookahead_pub.publish(point)
        self.closest_pub.publish(point2)
        
        # self.get_logger().info(str(p) + " -> " + str(closest) + " -> " + str(target))
                
        drive_cmd = AckermannDriveStamped()
        
        if t >= 1 or np.linalg.norm(p - self.spline.get(1)) < self.END_DIST:
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
        # startAngle = euler_from_quaternion([
        #     msg.poses[0].orientation.x,
        #     msg.poses[0].orientation.y,
        #     msg.poses[0].orientation.z,
        #     msg.poses[0].orientation.w
        # ])[2]
        
        # endAngle = euler_from_quaternion([
        #     msg.poses[-1].orientation.x,
        #     msg.poses[-1].orientation.y,
        #     msg.poses[-1].orientation.z,
        #     msg.poses[-1].orientation.w
        # ])[2]
        startAngle = 0
        endAngle = 0
        
        self.pp = []
        self.pp.append(self.points[0])
        for i in range(1, len(self.points)-5):
            if i % 5 == 0:
                self.pp.append(self.points[i])
        self.pp.append(self.points[-1])  
        self.spline = CubicHermiteGroup(self.pp, startAngle, endAngle)
        
        for p in self.pp:
            self.get_logger().info(str(p))

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
