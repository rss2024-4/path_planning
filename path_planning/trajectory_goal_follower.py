import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, Point, PointStamped, PoseArray, Pose
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from tf_transformations import euler_from_quaternion
from std_msgs.msg import String

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
U_TURNING = 3

RIGHT = 'RIGHT'
LEFT = 'LEFT'

# TODO: redo uturn to turn for a set number of seconds
# after uturning, reprocess self.default_points and self.default_visited- precompute the projections and sides in trajectory callback so this is faster
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
        self.center_segments = [] # segment_idx refer to the segments in this array

        self.centerline_trajectory = LineTrajectory(node=self, viz_namespace="/center_trajectory")
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
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                            #    self.drive_topic,
                                                "/follower_drive",
                                               1)
        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.odom_callback,
                                                 1)
        self.uturn_pub = self.create_publisher(String, "/uturn", 1)
        self.uturn_sub = self.create_subscription(String, "/uturn_is_done", self.uturn_done, 1)
        
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
        # Point_with_data is defined to be: (point, (side, segment_idx, projection_onto_segment))
        # points are tuples: (float, float)
        self.goal_points = [] # list of tuples: (point, (side, segment_idx, projection_onto_segment))
        self.trajectory_points = [] # list of tuples: (point, (side, segment_idx, projection_onto_segment))
        
        self.state = DEFAULT
        self.initialized_map = False
        self.goal_trajectory = LineTrajectory(node=self, viz_namespace="/goal_trajectory")

        self.start_wait = None
        self.previous_target = None

        self.turn_starte_time = 0.0

        self.uturn_point = None

    def load_centerline(self, path):
        path = os.path.expandvars(path)

        temp = []
        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                temp.append((p["x"], p["y"]))
                self.centerline.append(np.array((p["x"], p["y"])))

        for i in range(len(temp) - 1):
            point = (temp[i][0], temp[i][1])
            next_point = (temp[i+1][0], temp[i+1][1])
            self.center_segments.append((point, next_point))
        
        self.get_logger().info(f'center segments len {len(self.center_segments)}')
        self.centerline_trajectory.points = self.centerline
        self.centerline_trajectory.publish_viz()

    def uturn_done(self, msg):
        self.state == DEFAULT

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
        self.initialized_map = True
        self.get_logger().info("Map processed")
    
    def lineSegToPoint2(self, start, end, p):
            l = self.dist2(start, end)
            t = max(0, min(1, np.dot(p-start, end-start)/l))
            projection = start + t * (end-start)
            return self.dist2(p, projection), projection
    
    def in_front(self, a, b):
        _, a_idx, a_proj = self.centerline_side(a)
        _, b_idx, b_proj = self.centerline_side(b)

        if a_idx < b_idx:
            return False
        elif a_idx > b_idx:
            return True
        else:
            return a_proj > b_proj
    
    def centerline_side(self, point):
        p = np.array([[point[0]], [point[1]], [0]])
        min_dist = np.inf
        ans = None

        for i in range(len(self.centerline)-1):
            start, end = self.centerline[i], self.centerline[i+1]
            dist, _ = self.lineSegToPoint2(start, end, point)
            if dist < min_dist:
                min_dist = dist
                side, perp = self.get_side(point, np.array([np.array(start), np.array(end)]))
                pt_on_line = perp + p
                ans = (side, i, (pt_on_line[0,0], pt_on_line[1,0]))
        return ans

    def get_side(self, P, end_points):
        p0 = np.array([[end_points[0][0]], [end_points[0][1]], [0]])
        v = np.array([[end_points[1][0]-end_points[0][0]], [end_points[1][1]-end_points[0][1]], [0]])
        p = np.array([[P[0]], [P[1]], [0]])
        t = (v.T@(p-p0))/(v.T@v)
        perp = v*t + (p0 - p)
        side = np.cross(perp[:,0], v[:,0])
        if side[2] < 0:
            return RIGHT, perp
        return LEFT, perp

    def goal_cb(self, msg):
        p = (msg.point.x, msg.point.y)
        side, idx, proj = self.centerline_side(p)
        self.publish_point([proj], self.point_pub)
        self.get_logger().info(f'got goal point: {msg.point}, side: {side}, idx: {idx}, projection: {proj}')
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
        
        points = self.default_points
        visited = self.default_visited

        target = None

        if self.state == DEFAULT:
            minDist = float('inf')
            goal = None

            car_side, car_idx, car_projection = self.centerline_side(p)
            goal_side, goal_idx, goal_projection = self.goal_points[self.goal_idx][1]
            self.last_side = car_side

            if self.is_behind((goal_side, goal_idx, goal_projection), (car_side, car_idx, car_projection)):
                self.get_logger().info(f'have to uturn is behind {self.goal_points[self.goal_idx][1]}, carside {car_side}')
                # self.turn_start_time = self.get_time()
                self.uturn_point = p
                self.state = U_TURNING
                return

            # self.get_logger().info("here")
            # check if close to next goal point
            d = self.dist2(p, self.goal_points[self.goal_idx][0])
            if d < 12 and d < minDist:
                minDist = d
                goal, _ = self.goal_points[self.goal_idx]
                self.get_logger().info(f'goalidx, {self.goal_idx}')
            
            # if close, then check that car is the on the same side
            if goal is not None:
                self.get_logger().info(f'close to goal point: {goal}, dist {minDist}')

                if car_side == goal_side:
                    self.get_logger().info(f'car on same side as goal, go to it')
                    # stop
                    drive_cmd = AckermannDriveStamped()
                    drive_cmd.drive.speed = 0.0
                    self.drive_pub.publish(drive_cmd)

                    if minDist >= 2.5:
                        # plan a traj
                        traj = self.astar.plan(p, goal)
                        if traj is not None:
                            self.state = PLANNED_PATH
                            self.get_logger().info(f"Receiving new trajectory to goal pose {goal} points")

                            self.goal_trajectory.clear()
                            # self.goal_trajectory.fromPoseArray(msg)
                            self.goal_trajectory.points = traj
                            self.goal_trajectory.publish_viz()

                            # self.goal_path = np.array(self.goal_trajectory.points)
                            self.goal_path = [(p, None) for p in self.goal_trajectory.points]
                            self.goal_visited = np.array([False] * len(self.goal_trajectory.points))
                        else:
                            self.get_logger().info(f'could not get path to goal')
                    else:
                        self.state = PLANNED_PATH
                        self.goal_path = [(goal, None)]
                        self.goal_visited = [False]

                    if self.state == PLANNED_PATH:
                        self.get_logger().info(f'checking that no target points will be passed')
                        for i, (pt, _) in enumerate(self.default_points):
                            if self.default_visited[i]:
                                continue
                            _, pt_idx, pt_proj = self.centerline_side(pt)
                            if goal_idx > pt_idx:
                                self.default_visited[i] = True
                            elif goal_idx == pt_idx:
                                # self.get_logger().info(f'checking {i}, goal: {goal_idx}, pt: {pt_idx}')
                                # self.get_logger().info(f'checking {i}, goalp: {goal_projection}, ptp: {pt_proj}')
                                if self.dist2(self.centerline[goal_idx], goal_projection) > self.dist2(self.centerline[goal_idx], pt_proj):
                                    self.default_visited[i] = True
                            
                else:
                    self.get_logger().info(f'car not on same side as goal')

        # if im going towards a goal point, follow those points instead
        elif self.state == PLANNED_PATH:
            # self.get_logger().info(f'following planned')
            points = self.goal_path
            visited = self.goal_visited
        elif self.state == U_TURNING:
            # if self.get_time() - self.turn_start_time < 2.:
            # if self.last_side == self.centerline_side(p)[0]:
                
            #     drive_msg = self.create_drive_msg(1.5, 0.2)
            #     self.drive_pub.publish(drive_msg)
            #     return

            car_info = self.centerline_side(self.uturn_point)

            goal_across = self.uturn_point + (np.array(car_info[2]) - self.uturn_point)*1.5
            
            # self.get_logger().info(f'uturning to goal, {goal_across}')
            self.publish_point([goal_across], self.point_pub)

            # if self.dist2(goal_across, p) < 0.1:
            cur_info = self.centerline_side(p)
            if self.last_side == cur_info[0]:
                self.get_logger().info(f'uturning to goal, {goal_across}, cur info {cur_info[0]}')
                self.pure_pursuit(goal_across, p, theta)
                return

            # self.get_logger().info(f'reached uturn goal {p}, cur_side: {cur_info[0]} ')
            drive_cmd = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            self.drive_pub.publish(drive_cmd)
            
            for i, data in enumerate(self.default_points):
                # self.get_logger().info(f'checking, cur info: {cur_info} ')
                if self.is_behind(data[1], cur_info):
                    self.default_visited[i] = True
                else:
                    self.default_visited[i] = False
                self.state = DEFAULT
                    # self.get_logger().info('done turn calculations')
            return
        
        # find target
        # target = None
        for i in range(len(points)):
            if not visited[i]:
                if self.dist2(points[i][0], p) < self.lookahead ** 2:
                    visited[i] = True
                else:
                    target = points[i][0]
                    break
        
        # if this means im within lookahead distance if my goal point
        if self.state == PLANNED_PATH and target is None:
            if self.dist2(points[-1][0], p) < 0.15:
                # self.get_logger().info(f'reached goal point {p}')
                drive_cmd = AckermannDriveStamped()
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
                target = points[-1][0]
        
        # otherwise stop/ find the correct angle
        self.pure_pursuit(target, p, theta)

    
    def pure_pursuit(self, target, p, theta):
        drive_cmd = AckermannDriveStamped()
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
            self.publish_point([target], self.point_pub)
            
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

    def publish_point(self, pts, pub):
        # Construct a line
        msg = Marker()
        msg.type = Marker.POINTS
        msg.header.frame_id = "map"

        # Set the size and color
        msg.scale.x = 0.2
        msg.scale.y = 0.2
        msg.color.a = 1.
        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0

        points = []
        for p in pts:
            # Fill the line with the desired values
            pt = Point()
            pt.x = p[0]
            pt.y = p[1]
            # msg.points = [pt]
            points.append(pt)

        # msg.points.append(pt)
        msg.points = points
        # Publish the line
        pub.publish(msg)

    
    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz()

        # self.default_points = np.array(self.trajectory.points)
        self.default_points = []
        self.default_visited = np.array([False] * len(self.trajectory.points))

        # calculate what side of centerline for each point
        self.trajectory_points = []
        for p in self.trajectory.points:
            side, segment, projection_onto_center = self.centerline_side(p)
            self.trajectory_points.append((p, (side, segment, projection_onto_center)))
            self.default_points.append((p, (side, segment, projection_onto_center)))

        self.initialized_traj = True

    def is_behind(self, point_with_data, car_point):
        '''
        Return True if point_1 is behind point_2
        point_with_data is defined to be: (side, segment_idx, projection_onto_segment))
        '''
        car_side = car_point[0]

        segment_side, segment_idx, _ = point_with_data
        proj = np.array(point_with_data[2])

        car_segment_idx = car_point[1]
        car_proj = np.array(car_point[2])

        if car_side == RIGHT: # if right
            # if the segment is before then point is before
            if segment_idx < car_segment_idx:
                return True
            # if equal
            elif segment_idx == car_segment_idx:
                # try:
                p0, _ = self.center_segments[segment_idx]
                # except Exception as e:
                #     self.get_logger().info(f'except {e}, idx {segment_idx}, len {len(self.center_segments)}')
                p0 = np.array(p0)
                v1 = proj - p0
                v2 = car_proj - p0
                if v1.T @ v1 < v2.T @ v2: # if p1 projection closer to start p1 is behind p2
                    return True
                else:
                    return False
            else:
                return False
        elif car_side == LEFT:
            if segment_side == RIGHT:
                return True
            if segment_idx > car_segment_idx:
                return True
            elif segment_idx == car_segment_idx:
                p0, _ = self.center_segments[segment_idx]
                p0 = np.array(p0)
                v1 = proj - p0
                v2 = car_proj - p0
                if v1.T @ v1 > v2.T @ v2: # if p1 projection closer to end p1 is behind p2
                    return True
                else:
                    return False
            else:
                return False
        raise Exception("not a side")

    
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
    
    def create_drive_msg(self, vel, angle):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'map'
        cmd.drive.steering_angle = float(angle)
        cmd.drive.speed = float(vel)
        return cmd
    
    def get_time(self):
        return self.get_clock().now().to_msg().sec + (self.get_clock().now().to_msg().nanosec * (10**-9))


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuitWithTargets()
    rclpy.spin(follower)
    rclpy.shutdown()
