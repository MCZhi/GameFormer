import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
# from utils.cubic_spline_planner import Spline2D

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

def compute_direction_diff(ego_theta, target_theta):
    delta = np.abs(ego_theta - target_theta)
    delta = np.where(delta > np.pi, 2*np.pi - delta, delta)

    return delta

def depth_first_search(cur_lane, lanes, dist=0, threshold=100):
    """
    Perform depth first search over lane graph up to the threshold.
    Args:
        cur_lane: Starting lane_id
        lanes: raw lane data
        dist: Distance of the current path
        threshold: Threshold after which to stop the search
    Returns:
        lanes_to_return (list of list of integers): List of sequence of lane ids
    """
    if dist > threshold:
        return [[cur_lane]]
    else:
        traversed_lanes = []
        child_lanes = lanes[cur_lane].exit_lanes

        if child_lanes:
            for child in child_lanes:
                centerline = np.array([(map_point.x, map_point.y, map_point.z) for map_point in lanes[child].polyline])
                cl_length = centerline.shape[0]
                curr_lane_ids = depth_first_search(child, lanes, dist + cl_length, threshold)
                traversed_lanes.extend(curr_lane_ids)

        if len(traversed_lanes) == 0:
            return [[cur_lane]]

        lanes_to_return = []

        for lane_seq in traversed_lanes:
            lanes_to_return.append([cur_lane] + lane_seq)
                
        return lanes_to_return

def is_overlapping_lane_seq(lane_seq1, lane_seq2):
    """
    Check if the 2 lane sequences are overlapping.
    Args:
        lane_seq1: list of lane ids
        lane_seq2: list of lane ids
    Returns:
        bool, True if the lane sequences overlap
    """

    if lane_seq2[1:] == lane_seq1[1:]:
        return True
    elif set(lane_seq2) <= set(lane_seq1):
        return True

    return False

def remove_overlapping_lane_seq(lane_seqs):
    """
    Remove lane sequences which are overlapping to some extent
    Args:
        lane_seqs (list of list of integers): List of list of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
    Returns:
        List of sequence of lane ids (e.g. ``[[12345, 12346, 12347], [12345, 12348]]``)
    """
    redundant_lane_idx = set()

    for i in range(len(lane_seqs)):
        for j in range(len(lane_seqs)):
            if i in redundant_lane_idx or i == j:
                continue
            if is_overlapping_lane_seq(lane_seqs[i], lane_seqs[j]):
                redundant_lane_idx.add(j)

    unique_lane_seqs = [lane_seqs[i] for i in range(len(lane_seqs)) if i not in redundant_lane_idx]
    
    return unique_lane_seqs

def polygon_completion(polygon):
    polyline_x = []
    polyline_y = []

    for i in range(len(polygon)):
        if i+1 < len(polygon):
            next = i+1
        else:
            next = 0

        dist_x = polygon[next].x - polygon[i].x
        dist_y = polygon[next].y - polygon[i].y
        dist = np.linalg.norm([dist_x, dist_y])
        interp_num = np.ceil(dist)*2
        interp_index = np.arange(2+interp_num)
        point_x = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].x, polygon[next].x]).tolist()
        point_y = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].y, polygon[next].y]).tolist()
        polyline_x.extend(point_x[:-1])
        polyline_y.extend(point_y[:-1])
    
    polyline_x, polyline_y = np.array(polyline_x), np.array(polyline_y)
    polyline_heading = wrap_to_pi(np.arctan2(polyline_y[1:]-polyline_y[:-1], polyline_x[1:]-polyline_x[:-1]))
    polyline_heading = np.insert(polyline_heading, -1, polyline_heading[-1])

    return np.stack([polyline_x, polyline_y, polyline_heading], axis=1)

def get_polylines(lines):
    polylines = {}

    for line in lines.keys():
        polyline = np.array([(map_point.x, map_point.y) for map_point in lines[line].polyline])
        if len(polyline) > 1:
            direction = wrap_to_pi(np.arctan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0]))
            direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
        else:
            direction = np.array([0])[:, np.newaxis]
        polylines[line] = np.concatenate([polyline, direction], axis=-1)

    return polylines

def find_reference_lanes(agent_type, agent_traj, lanes):
    curr_lane_ids = {}
        
    if agent_type == 2:
        distance_threshold = 5

        while len(curr_lane_ids) < 1:
            for lane in lanes.keys():
                if lanes[lane].shape[0] > 1:
                    distance_to_agent = LineString(lanes[lane][:, :2]).distance(Point(agent_traj[-1, :2]))
                    if distance_to_agent < distance_threshold:
                        curr_lane_ids[lane] = 0          

            distance_threshold += 5
            if distance_threshold > 50:
                break
    else:
        distance_threshold = 3.5
        direction_threshold = 10
        while len(curr_lane_ids) < 1:
            for lane in lanes.keys():
                distance_to_ego = np.linalg.norm(agent_traj[-1, :2] - lanes[lane][:, :2], axis=-1)
                direction_to_ego = compute_direction_diff(agent_traj[-1, 2], lanes[lane][:, -1])
                for i, j, k in zip(distance_to_ego, direction_to_ego, range(distance_to_ego.shape[0])):    
                    if i <= distance_threshold:# and j <= np.radians(direction_threshold):
                        curr_lane_ids[lane] = k
                        break
            
            distance_threshold += 3.5
            direction_threshold += 10  
            if distance_threshold > 50:
                break

    return curr_lane_ids

def find_neighbor_lanes(curr_lane_ids, traj, lanes, lane_polylines):
    neighbor_lane_ids = {}

    for curr_lane, start in curr_lane_ids.items():
        left_lanes = lanes[curr_lane].left_neighbors
        right_lanes = lanes[curr_lane].right_neighbors
        left_lane = None
        right_lane = None
        curr_index = start

        for l_lane in left_lanes:
            if l_lane.self_start_index <= curr_index <= l_lane.self_end_index and not l_lane.feature_id in curr_lane_ids:
                left_lane = l_lane

        for r_lane in right_lanes:
            if r_lane.self_start_index <= curr_index <= r_lane.self_end_index and not r_lane.feature_id in curr_lane_ids:
                right_lane = r_lane

        if left_lane is not None:    
            left_polyline = lane_polylines[left_lane.feature_id]
            start = np.argmin(np.linalg.norm(traj[-1, :2] - left_polyline[:, :2], axis=-1))
            neighbor_lane_ids[left_lane.feature_id] = start              

        if right_lane is not None:
            right_polyline = lane_polylines[right_lane.feature_id]
            start = np.argmin(np.linalg.norm(traj[-1, :2] - right_polyline[:, :2], axis=-1)) 
            neighbor_lane_ids[right_lane.feature_id] = start

    return neighbor_lane_ids

def find_neareast_point(curr_point, line):
    distance_to_curr_point = np.linalg.norm(curr_point[np.newaxis, :2] - line[:, :2], axis=-1)
    neareast_point = line[np.argmin(distance_to_curr_point)]
    
    return neareast_point


def imputer(traj):
    x, y, v_x, v_y, theta = traj[:, 0], traj[:, 1], traj[:, 3], traj[:, 4], traj[:, 2]

    if np.any(x==0):
        for i in reversed(range(traj.shape[0])):
            if x[i] == 0:
                v_x[i] = v_x[i+1]
                v_y[i] = v_y[i+1]
                x[i] = x[i+1] - v_x[i]*0.1
                y[i] = y[i+1] - v_y[i]*0.1
                theta[i] = theta[i+1]
        return np.column_stack((x, y, theta, v_x, v_y))
    else:
        return np.column_stack((x, y, theta, v_x, v_y))

def agent_norm(traj, center, angle, impute=False):
    if impute:
        traj = imputer(traj[:, :5])

    line = LineString(traj[:, :2])
    line_offset = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
    line_rotate = rotate(line_offset, -angle, origin=(0, 0), use_radians=True)
    line_rotate = np.array(line_rotate.coords)
    line_rotate[traj[:, :2]==0] = 0

    heading = wrap_to_pi(traj[:, 2] - angle)
    heading[traj[:, 2]==0] = 0

    if traj.shape[-1] > 3:
        velocity_x = traj[:, 3] * np.cos(angle) + traj[:, 4] * np.sin(angle)
        velocity_x[traj[:, 3]==0] = 0
        velocity_y = traj[:, 4] * np.cos(angle) - traj[:, 3] * np.sin(angle)
        velocity_y[traj[:, 4]==0] = 0
        return np.column_stack((line_rotate, heading, velocity_x, velocity_y))
    else:
        return  np.column_stack((line_rotate, heading))

def map_norm(map_line, center, angle):
    self_line = LineString(map_line[:, 0:2])
    self_line = affine_transform(self_line, [1, 0, 0, 1, -center[0], -center[1]])
    self_line = rotate(self_line, -angle, origin=(0, 0), use_radians=True)
    self_line = np.array(self_line.coords)
    self_line[map_line[:, 0:2]==0] = 0
    self_heading = wrap_to_pi(map_line[:, 2] - angle)

    if map_line.shape[1] > 3:
        left_line = LineString(map_line[:, 3:5])
        left_line = affine_transform(left_line, [1, 0, 0, 1, -center[0], -center[1]])
        left_line = rotate(left_line, -angle, origin=(0, 0), use_radians=True)
        left_line = np.array(left_line.coords)
        left_line[map_line[:, 3:5]==0] = 0
        left_heading = wrap_to_pi(map_line[:, 5] - angle)
        left_heading[map_line[:, 5]==0] = 0

        right_line = LineString(map_line[:, 6:8])
        right_line = affine_transform(right_line, [1, 0, 0, 1, -center[0], -center[1]])
        right_line = rotate(right_line, -angle, origin=(0, 0), use_radians=True)
        right_line = np.array(right_line.coords)
        right_line[map_line[:, 6:8]==0] = 0
        right_heading = wrap_to_pi(map_line[:, 8] - angle)
        right_heading[map_line[:, 8]==0] = 0

        return np.column_stack((self_line, self_heading, left_line, left_heading, right_line, right_heading))
    else:
        return np.column_stack((self_line, self_heading))