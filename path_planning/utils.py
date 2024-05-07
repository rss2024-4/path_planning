import rclpy

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import Header
import os
from typing import List, Tuple
import json

EPSILON = 0.00000000001

''' These data structures can be used in the search function
'''

# class CubicHermite:
#     def __init__(self, p0, p1, v0, v1):
#         self.p0 = p0
#         self.p1 = p1
#         self.v0 = v0
#         self.v1 = v1
        
#     def get(self, t):
#         return (2 * t**3 - 3 * t**2 + 1) * self.p0 + \
#                     (t**3 - 2 * t**2 + t) * self.v0 + \
#                     (-2 * t**3 + 3 * t**2) * self.p1 + \
#                     (t**3 - t**2) * self.v1


class CubicHermite():
    def __init__(self, pos0, pos1, vel0, vel1):
        self.pos0 = pos0
        self.pos1 = pos1
        self.vel0 = vel0
        self.vel1 = vel1
        self.length = self.getGaussianQuadratureLength(0, 1)

    def get(self, t, nD=0):
        if t < 0:
            return self.pos0
        if t > 1:
            return self.pos1
        return self.pos0 * self.basis(t, 0, nD) + \
                    self.pos1 * self.basis(t, 3, nD) + \
                    self.vel0 * self.basis(t, 1, nD) + \
                    self.vel1 * self.basis(t, 2, nD)

    def basis(self, t, i, nD):
        if nD==0:
            if i==0:
                return 1 - 3 * t * t + 2 * t * t * t
            elif i==1:
                return t - 2 * t * t + t * t * t
            elif i==2:
                return -t * t + t * t * t
            elif i==3:
                return 3 * t * t - 2 * t * t * t
            else:
                return 0
        elif nD==1:
            if i==0:
                return -6 * t + 6 * t * t
            elif i==1:
                return 1 - 4 * t + 3 * t * t
            elif i==2:
                return -2 * t + 3 * t * t
            elif i==3:
                return 6 * t - 6 * t * t
            else:
                return 0
        elif nD==2:
            if i==0:
                return -6 + 12 * t
            elif i==1:
                return -4 + 6 * t
            elif i==2:
                return -2 + 6 * t
            elif i==3:
                return 6 - 12 * t
            else:
                return 0
        else:
            return 0

    def getGaussianCoefs(self):
        return [
            [0.179446470356207, 0.0000000000000000],
            [0.176562705366993, -0.178484181495848],
            [0.176562705366993, 0.178484181495848],
            [0.1680041021564500, -0.351231763453876],
            [0.1680041021564500, 0.351231763453876],
            [0.15404576107681, -0.512690537086477],
            [0.15404576107681, 0.512690537086477],
            [0.135136368468526, -0.657671159216691],
            [0.135136368468526, 0.657671159216691],
            [0.1118838471934040, -0.781514003896801],
            [0.1118838471934040, 0.781514003896801],
            [0.0850361483171792, -0.880239153726986],
            [0.0850361483171792, 0.880239153726986],
            [0.0554595293739872, -0.950675521768768],
            [0.0554595293739872, 0.950675521768768],
            [0.0241483028685479, -0.990575475314417],
            [0.0241483028685479, 0.990575475314417]
        ]

    def getGaussianQuadratureLength(self, start, end):
        coefficients = self.getGaussianCoefs()
        half = (end - start) / 2.0
        avg = (start + end) / 2.0
        length = 0
        for coefficient in coefficients:
            length += np.linalg.norm(self.get(((avg + half * coefficient[1])), 1)) * coefficient[0]
        return length * half

    def getClosestPoint(self, point):
        return self.get(self.findClosestPointOnSpline(point))

    def findClosestPointOnSpline(self, point, steps=10, iterations=5):
        cur_dist = float("inf")
        cur_min = 0

        i = 0
        while i <= 1:
            cur_t = i
            
            i += 1. / steps
            
            if self.getSecondDerivAtT(cur_t, point) == 0: dt = 0
            else: dt = self.getFirstDerivAtT(cur_t, point) / self.getSecondDerivAtT(cur_t, point)
            counter = 0
            
            while dt != 0 and counter < iterations:
                # adjust based on Newton's method, get new derivatives
                cur_t -= dt
                if self.getSecondDerivAtT(cur_t, point) == 0: dt = 0
                else: dt = self.getFirstDerivAtT(cur_t, point) / self.getSecondDerivAtT(cur_t, point)
                counter += 1
                                
            cur_d = (self.get(cur_t)[0] - point[0])**2 + (self.get(cur_t)[1] - point[1])**2
            # if distance is less than previous min, update distance and t
            if cur_d < cur_dist:
                cur_dist = cur_d
                cur_min = cur_t
            i += 1. / steps
        return (min(1, max(0, cur_min)))

    def getFirstDerivAtT(self, t, point):
        p = self.get(t)
        d1 = self.get(t, 1)
        x_a = p[0] - point[0]
        y_a = p[1] - point[1]
        return 2 * (x_a * d1[0] + y_a * d1[1])

    def getSecondDerivAtT(self, t, point):
        p = self.get(t)
        d1 = self.get(t, 1)
        d2 = self.get(t, 2)
        x_a = p[0] - point[0]
        y_a = p[1] - point[1]
        return 2 * (d1[0] * d1[0] + x_a * d2[0] + d1[1] * d1[1] + y_a * d2[1])

    def getTFromLength(self, length):
        t = length / self.length
       
        i = 0
        while i < 5:
            derivativeMagnitude = np.linalg.norm(self.get(t, 1))
            if derivativeMagnitude > 0.0:
                t -= (self.getGaussianQuadratureLength(0, t) - length) / derivativeMagnitude
                # Clamp to [0, 1]
                t = min(1, max(t, 0))
            i += 1
        return t
    
    
    


class CubicHermiteGroup:
    def __init__(self, points, startAngle, endAngle, angleWeight=1):
        self.points = points
        self.angles = [startAngle, endAngle]
        self.angleWeight = angleWeight
        self.splines = []
        self.splines_l = []
        self.n = len(self.points)
        self.gen_splines()
        
    def gen_splines(self):
        for i in range(self.n-1):
            p0 = self.points[i]
            p1 = self.points[i+1]
            if i == 0:
                # v0 = self.angleToVec(self.angles[0])
                v0 = self.points[i+1] - self.points[i]
                v1 = self.points[i+2] - self.points[i]
            elif i == self.n-2:
                v0 = self.points[i+1] - self.points[i-1]
                # v1 = self.angleToVec(self.angles[1])
                v1 = self.points[i+1] - self.points[i]
            else:
                v0 = self.points[i+1] - self.points[i-1]
                v1 = self.points[i+2] - self.points[i]
                
            v0 = v0 / np.linalg.norm(v0) * self.angleWeight
            v1 = v1 / np.linalg.norm(v1) * self.angleWeight
                
            self.splines.append(CubicHermite(p0, p1, v0, v1))
            self.splines_l.append(self.splines[i].length)
            
    def angleToVec(self, angle):
        return np.array([np.cos(angle), np.sin(angle)])
            
    def getIndexFromT(self, t):
        if t >= 1: return self.n-2
        if t <= 0: return 0
        return int(t * (self.n - 1))
    
    def getSubTFromT(self, t):
        if t >= 1: return 1
        if t <= 0: return 0
        return t * (self.n - 1) - self.getIndexFromT(t)
    
    def subTAndIndexToT(self, n, sT):
        return (n + sT) / (self.n - 1)
    
    def get(self, t):
        return self.splines[self.getIndexFromT(t)].get(self.getSubTFromT(t))
    
    def getEndVel(self):
        return self.splines[-1].vel1
        
    def getClosestPointT(self, point):
        cur_min_dist = float('inf')
        cur_min = None
        
        for i, s in enumerate(self.splines):
            t = s.findClosestPointOnSpline(point)
            p = s.get(t)
            dist = np.linalg.norm(p - point)
            if dist < cur_min_dist:
                cur_min_dist = dist
                cur_min = (i, t)
                
        return self.subTAndIndexToT(cur_min[0], cur_min[1])
    
    def getLengthFromT(self, t):
        ind = self.getIndexFromT(t)
        return sum(self.splines_l[:ind]) + self.splines[ind].getGaussianQuadratureLength(0, self.getSubTFromT(t))
    
    def getTFromLength(self, l):
        sm = 0
        t = 0
        for s in self.splines:
            if sm + s.length > l:
                return t + s.getTFromLength(l - sm) / (self.n-1)
            else:
                sm += s.length
                t += 1 / (self.n-1)
                
        return t

    def getLookaheadPointFromT(self, t, point, lookahead):
        l = self.getLengthFromT(t)
        if t == 0:
            dist = np.linalg.norm(point - self.get(0))
            # if dist > lookahead:
            #     return # line case
            # else:
            return self.get(self.getTFromLength(max(0, lookahead - dist)))
        elif l + lookahead > sum(self.splines_l):
            dist = l + lookahead - sum(self.splines_l)
            return self.get(1) + dist * (self.getEndVel() / np.linalg.norm(self.getEndVel()))
        return self.get(self.getTFromLength(l + lookahead))
        
    def getLookaheadPoint(self, point, lookahead):
        return self.getLookaheadPointFromT(self.getClosestPointT(point), point, lookahead)
            
        

class LineTrajectory:
    """ A class to wrap and work with piecewise linear trajectories. """

    def __init__(self, node, viz_namespace=None):
        self.points = []
        self.distances = []
        self.has_acceleration = False
        self.visualize = False
        self.viz_namespace = viz_namespace
        self.node = node

        if viz_namespace:
            self.visualize = True
            self.start_pub = self.node.create_publisher(Marker, viz_namespace + "/start_point", 1)
            self.traj_pub = self.node.create_publisher(Marker, viz_namespace + "/path", 1)
            self.end_pub = self.node.create_publisher(Marker, viz_namespace + "/end_pose", 1)

    # compute the distances along the path for all path segments beyond those already computed
    def update_distances(self):
        num_distances = len(self.distances)
        num_points = len(self.points)

        for i in range(num_distances, num_points):
            if i == 0:
                self.distances.append(0)
            else:
                p0 = self.points[i - 1]
                p1 = self.points[i]
                delta = np.array([p0[0] - p1[0], p0[1] - p1[1]])
                self.distances.append(self.distances[i - 1] + np.linalg.norm(delta))

    def distance_to_end(self, t):
        if not len(self.points) == len(self.distances):
            print(
                "WARNING: Different number of distances and points, this should never happen! Expect incorrect results. See LineTrajectory class.")
        dat = self.distance_along_trajectory(t)
        if dat == None:
            return None
        else:
            return self.distances[-1] - dat

    def distance_along_trajectory(self, t):
        # compute distance along path
        # ensure path boundaries are respected
        if t < 0 or t > len(self.points) - 1.0:
            return None
        i = int(t)  # which segment
        t = t % 1.0  # how far along segment
        if t < EPSILON:
            return self.distances[i]
        else:
            return (1.0 - t) * self.distances[i] + t * self.distances[i + 1]

    def addPoint(self, point: Tuple[float, float]) -> None:
        print("adding point to trajectory:", point)
        self.points.append(point)
        self.update_distances()
        self.mark_dirty()

    def clear(self):
        self.points = []
        self.distances = []
        self.mark_dirty()

    def empty(self):
        return len(self.points) == 0

    def save(self, path):
        print("Saving trajectory to:", path)
        data = {}
        data["points"] = []
        for p in self.points:
            data["points"].append({"x": p[0], "y": p[1]})
        with open(path, 'w') as outfile:
            json.dump(data, outfile)

    def mark_dirty(self):
        self.has_acceleration = False

    def dirty(self):
        return not self.has_acceleration

    def load(self, path):
        print("Loading trajectory:", path)

        # resolve all env variables in path
        path = os.path.expandvars(path)

        with open(path) as json_file:
            json_data = json.load(json_file)
            for p in json_data["points"]:
                self.points.append((p["x"], p["y"]))
        self.update_distances()
        print("Loaded:", len(self.points), "points")
        self.mark_dirty()

    # build a trajectory class instance from a trajectory message
    def fromPoseArray(self, trajMsg):
        for p in trajMsg.poses:
            self.points.append((p.position.x, p.position.y))
        self.update_distances()
        self.mark_dirty()
        print("Loaded new trajectory with:", len(self.points), "points")
        
    def fromPointArray(self, arr):
        for p in arr:
            self.points.append((p[0], p[1]))
        self.update_distances()
        self.mark_dirty()

    def toPoseArray(self):
        traj = PoseArray()
        traj.header = self.make_header("/map")
        for i in range(len(self.points)):
            p = self.points[i]
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            traj.poses.append(pose)
        return traj

    def publish_start_point(self, duration=0.0, scale=0.1):
        should_publish = len(self.points) > 0
        self.node.get_logger().info("Before Publishing start point")
        if self.visualize and self.start_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing start point")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 0
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[0][0]
                marker.pose.position.y = self.points[0][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.start_pub.publish(marker)
        elif self.start_pub.get_subscription_count() == 0:
            self.node.get_logger().info("Not publishing start point, no subscribers")

    def publish_end_point(self, duration=0.0):
        should_publish = len(self.points) > 1
        if self.visualize and self.end_pub.get_subscription_count() > 0:
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 1
            marker.type = 2  # sphere
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                marker.action = 0
                marker.pose.position.x = self.points[-1][0]
                marker.pose.position.y = self.points[-1][1]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # delete marker
                marker.action = 2

            self.end_pub.publish(marker)
        elif self.end_pub.get_subscription_count() == 0:
            print("Not publishing end point, no subscribers")

    def publish_trajectory(self, duration=0.0):
        should_publish = len(self.points) > 1
        self.node.get_logger().info(f'Should publish traj: {should_publish}')
        if self.visualize and self.traj_pub.get_subscription_count() > 0:
            self.node.get_logger().info("Publishing trajectory")
            marker = Marker()
            marker.header = self.make_header("/map")
            marker.ns = self.viz_namespace + "/trajectory"
            marker.id = 2
            marker.type = marker.LINE_STRIP  # line strip
            marker.lifetime = rclpy.duration.Duration(seconds=duration).to_msg()
            if should_publish:
                # marker.action = marker.ADD
                # marker.points = [Point(p[0], p[1]) for p in self.points]
                new_points = []
                for p in self.points:
                    pt = Point()
                    pt.x = p[0]
                    pt.y = p[1]
                    pt.z = 0.0
                    new_points.append(pt)
                
                marker.points = new_points
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                # for p in self.points:
                #     pt = Point()
                #     pt.x = p[0]
                #     pt.y = p[1]
                #     pt.z = 0.0
                #     marker.points.append(pt)
            else:
                # delete
                marker.action = marker.DELETE
            self.traj_pub.publish(marker)
            print('publishing traj')
        elif self.traj_pub.get_subscription_count() == 0:
            print("Not publishing trajectory, no subscribers")

    def publish_viz(self, duration=0):
        if not self.visualize:
            print("Cannot visualize path, not initialized with visualization enabled")
            return
        self.publish_start_point(duration=duration)
        self.publish_trajectory(duration=duration)
        self.publish_end_point(duration=duration)

    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = self.node.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header
