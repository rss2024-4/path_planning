import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

import numpy as np
import time

from .utils import LineTrajectory


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
        
        # self.test_find_closest_point_on_trajectory()
        self.start_time = None
        self.end_time = None


        # for testing
        # self.initialized_traj = True
        # self.points = [(-9.25, 26.0), (-9.0, 25.75), (-8.75, 25.75), (-8.5, 25.5), (-8.25, 25.25), (-8.0, 25.25), (-7.75, 25.25), (-7.5, 25.25), (-7.25, 25.25), (-7.25, 25.0), (-7.25, 24.75), (-7.0, 24.5), (-7.0, 24.25), (-7.0, 24.0), (-7.0, 23.75), (-7.0, 23.5), (-7.0, 23.25), (-7.0, 23.0), (-6.75, 22.75), (-6.5, 22.5), (-6.5, 22.25), (-6.5, 22.0), (-6.5, 21.75), (-6.5, 21.5), (-6.5, 21.25), (-6.75, 21.0), (-7.0, 20.75), (-7.25, 20.5), (-7.5, 20.25), (-7.75, 20.0), (-8.0, 19.75), (-8.25, 19.5), (-8.5, 19.25), (-8.75, 19.0), (-9.0, 18.75), (-9.25, 18.5), (-9.5, 18.25), (-9.75, 18.0), (-10.0, 17.75), (-10.25, 17.5), (-10.5, 17.25), (-10.75, 17.0), (-11.0, 16.75), (-11.25, 16.5), (-11.5, 16.25), (-11.75, 16.0), (-12.0, 15.75), (-12.25, 15.5), (-12.5, 15.25), (-12.75, 15.0), (-13.0, 14.75), (-13.25, 14.5), (-13.5, 14.25), (-13.75, 14.0), (-14.0, 13.75), (-14.25, 13.5), (-14.5, 13.25), (-14.75, 13.0), (-15.0, 12.75), (-15.0, 12.5), (-15.25, 12.25), (-15.5, 12.0), (-15.75, 11.75), (-16.0, 11.5), (-16.0, 11.25), (-16.0, 11.0), (-16.0, 10.75), (-16.0, 10.5), (-16.0, 10.25), (-16.0, 10.0), (-16.0, 9.75), (-16.25, 9.75), (-16.25, 9.5), (-16.5, 9.5), (-16.75, 9.25), (-17.0, 9.0), (-17.25, 8.75), (-17.5, 8.5), (-17.75, 8.25), (-18.0, 8.0), (-18.25, 7.75), (-18.5, 7.5), (-18.75, 7.25), (-19.0, 7.0), (-19.25, 6.75), (-19.5, 6.5), (-19.75, 6.25), (-20.0, 6.0), (-20.25, 5.75), (-20.25, 5.5), (-20.25, 5.25), (-20.5, 5.0), (-20.5, 4.75), (-20.5, 4.5), (-20.5, 4.25), (-20.75, 4.0), (-20.75, 3.75), (-21.0, 3.5), (-21.0, 3.25), (-21.25, 3.0), (-21.25, 2.75), (-21.25, 2.5), (-21.5, 2.25), (-21.5, 2.0), (-21.75, 1.75), (-21.75, 1.5), (-21.75, 1.25), (-22.0, 1.0), (-22.0, 0.75), (-22.25, 0.5), (-22.5, 0.5), (-22.75, 0.5), (-23.0, 0.25), (-23.25, 0.0), (-23.5, 0.0), (-23.75, -0.25), (-24.0, -0.25)]
        # self.visited = np.array([False] * len(self.points))

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
            drive_cmd.drive.speed = 0.0
            if not self.end_time:
                self.end_time = time.time()
                elapsed = time.time() - self.start_time
                self.get_logger().info(f'Elapsed: {elapsed}')
                self.start_time = None
        else:
            angle = self.find_steering_angle(p, theta, target)
            if np.abs(angle) < 0.05:
                drive_cmd.drive.speed = 2.0
            else:
                drive_cmd.drive.speed = self.speed*1.0
            drive_cmd.drive.steering_angle = angle

            if not self.start_time:
                self.get_logger().info("started")
                self.start_time = time.time()
            
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
