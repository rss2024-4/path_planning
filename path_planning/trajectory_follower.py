import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, Point
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

import numpy as np

from .utils import LineTrajectory
from visualization_msgs.msg import Marker


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        # self.drive_topic = "/vesc/low_level/input/navigation"
        self.driver_topic = "/drive"
        self.get_logger().info(f'{self.drive_topic}')

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
        
        self.point_pub = self.create_publisher(Marker, "lookahead", 1)
        # self.test_find_closest_point_on_trajectory()

    def dist2(self, p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1] - p2[1])**2
    
    def find_closest_point(self, p): # p is current position
        minDist = None
        closestPoint = None
        closestIdx = None

        for i in range(len(self.points)-1):
            start, end = self.points[i], self.points[i+1]
            dist, projection = self.lineSegToPoint2(start, end, p)
            # print('dist', i, dist)
            if not minDist or dist < minDist:
                minDist = dist
                closestPoint = projection
                closestIdx = i
            
        return minDist, closestIdx


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
        
        # self.get_logger().info(str(self.visited))

        target = None
        for i in range(len(self.points)):
            if not self.visited[i]:
                if self.dist2(self.points[i], p) < self.lookahead ** 2:
                    self.visited[i] = True
                else:
                    target = self.points[i]
                    break
                
        drive_cmd = AckermannDriveStamped()
        
        if target is None:
            self.get_logger().info("no target")
            drive_cmd.drive.speed = 0.0
        else:
            
            angle = self.find_steering_angle(p, theta, target)
            if np.isnan(angle):
                self.get_logger().info("null angle")
                angle = 0.0
            if np.abs(angle) < 0.05:
                drive_cmd.drive.speed = 2.0
            else:
                drive_cmd.drive.speed = self.speed*1.0

            self.get_logger().info(f'angle: {angle}, speed: {drive_cmd.drive.speed}, target: {target}')
            drive_cmd.drive.steering_angle = angle
            self.publish_point(target)
            
        self.drive_pub.publish(drive_cmd)
            

    def find_steering_angle(self, p, theta, lookaheadPoint):
        target = lookaheadPoint - p
        car_vec = np.cos(theta), np.sin(theta) # vector of car
        
        # steer
        d = np.linalg.norm(target)
        eta = np.arccos(np.dot(car_vec, target)/(d * np.linalg.norm(car_vec)))
        # self.get_logger().info(f'{eta}, {np.dot(car_vec, target)}, {np.linalg.norm(car_vec)}')
        delta = np.arctan(2*self.wheelbase_length*np.sin(eta))
        sign = np.sign(np.cross(car_vec, target))
        return delta*sign
        

    def find_lookahead(self, p, closestIdx):
        points_ahead = np.array(self.points[closestIdx:])
        intersections = []
        for i in range(len(points_ahead)-1):
            start, end = points_ahead[i], points_ahead[i+1]
            V = end-start
            a = V.dot(V)
            b = 2 * V.dot(start - p)
            c = start.dot(start) + p.dot(p) - 2 * start.dot(p) - self.lookahead**2


            # discriminant
            disc = b**2 - 4 * a * c
            if disc < 0:
                continue
            else:
                self.get_logger().info('found point in circle')
                sqrt_disc = np.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)

                if 0 <= t1 <= 1 and 0 <= t2 <= 1:   # both valid
                    t = max(t1, t2)
                elif 0 <= t1 <= 1:
                    t = t1
                elif 0 <= t2 <= 1:
                    t = t2
                else:
                    # self.get_logger().info("invalid solutions")
                    continue
                intersections.append(start + t * V)
        if not intersections:
            self.get_logger().info('intersections no exist')
            return None
        return intersections[-1]

    def test_find_closest_point_on_trajectory(self):
        print("Testing find_closest_point_on_trajectory")
        # current_pose = [0, 0, 0]
        # self.trajectory.points = [[0, 1], [1, 1], [2, 20]]
        # closest_index = self.find_closest_point_on_trajectory(current_pose)
        # assert closest_index == 0, "Closest index should be 0, got %d" % closest_index
        current_pose = (0, 0)
        self.points = np.array([[0, 100], [0, 50], [3, 50], [10, 50], [1, 1]])
        closest_index = self.find_closest_point(current_pose)
        print('closest index', closest_index)
        assert closest_index[1] == 3, "Closest index should be 3, got %d" % closest_index[1]
        print("test_find_closest_point_on_trajectory..........OK!")

    # def find_closest_point(self, cur_pos):
        
    # def find_lookahead_point(self, closest)
    
    def pose_callback_fast(self, msg):
        pass

    # start, end, current are points
    def lineSegToPoint2(self, start, end, p):
        l = self.dist2(start, end)
        t = max(0, min(1, np.dot(p-start, end-start)/l))
        # t = max(0, min(1, np.dot((p[0]-start[0], p[1]-start[1]), (end[0]-start[0], end[1]-start[1])) / l))
        # projection = (start[0]*(1-t) + t * end[0], start[1]*(1-t) + end[1]*t)
        projection = start + t * (end-start)
        return self.dist2(p, projection), projection



    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz()

        self.points = np.array(self.trajectory.points)
        self.visited = np.array([False] * len(self.trajectory.points))
        # self.get_logger().info(f'Points: {",".join(self.trajectory.points)}')
        #for p in self.points:
        #    self.get_logger().info(str(p))

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
