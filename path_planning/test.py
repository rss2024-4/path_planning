import numpy as np
import math



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

    def findClosestPointOnSpline(self, point, steps=15, iterations=8):
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
                v0 = self.angleToVec(self.angles[0])
                v1 = self.points[i+2] - self.points[i]
            elif i == self.n-2:
                v0 = self.points[i+1] - self.points[i-1]
                v1 = self.angleToVec(self.angles[1])
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

    def getLookaheadPoint(self, point, lookahead):
        t = self.getClosestPointT(point)
        l = self.getLengthFromT(t)
        if t == 0:
            dist = np.linalg.norm(point - self.get(0))
            # if dist > lookahead:
            #     return # line case
            # else:
            return self.get(self.getTFromLength(max(0, lookahead - dist)))
        elif l + lookahead > sum(self.splines_l):
            dist = l + lookahead - sum(self.splines_l)
            return self.get(1) + dist * np.array([np.cos(self.angles[1]), np.sin(self.angles[1])])
        return self.get(self.getTFromLength(l + lookahead))
            
        
    
            
s = CubicHermiteGroup(np.array([[0, 0], [1, 1], [2, 0], [3, 2]]), 0, 0)

print(s.getLookaheadPoint(np.array([1, 1]), 1))
# for i in range(10 + 1):
#     print(i/10, s.getIndexFromT(i / 10), s.getSubTFromT(i / 10))



# st = []
# for i in range(100 + 1):
#     p = s.get(i / 100)
#     st.append("(%.5f, %.5f)" % (p[0], p[1]))
    
# print("[" + ", ".join(st) + "]")

# t = s.getClosestPointT(np.array([0.5, 1]))
# p = s.get(t)

# print(p, s.getLookaheadPoint(np.array([0.5, 1]), 0.2))
# print(s.spline_l)
# print(s.getLengthFromT(t))
