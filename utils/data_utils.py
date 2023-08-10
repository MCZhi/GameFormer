import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from utils.cubic_spline_planner import Spline2D


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


def generate_target_course(x, y):
    csp = Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def find_map_waypoint(pos, polylines):
    waypoint = [-1, -1, 1e9, 1e9]
    direction_threshold = 10

    for id, polyline in polylines.items():
        distance_to_gt = np.linalg.norm(pos[np.newaxis, :2] - polyline[:, :2], axis=-1)
        direction_to_gt = compute_direction_diff(pos[np.newaxis, 2], polyline[:, 2])
        for i, j, k in zip(range(polyline.shape[0]), distance_to_gt, direction_to_gt):    
            if j < waypoint[2] and k <= np.radians(direction_threshold):
                waypoint = [id, i, j, k]
    
    lane_id = waypoint[0]
    waypoint_id = waypoint[1]

    if lane_id > 0:
        return lane_id, waypoint_id
    else:
        return None, None


def find_route(traj, timestep, cur_pos, map_lanes, map_signals):
    lane_polylines = get_polylines(map_lanes)
    end_lane, end_point = find_map_waypoint(np.array((traj[-1].center_x, traj[-1].center_y, traj[-1].heading)), lane_polylines)
    cur_lane, _ = find_map_waypoint(cur_pos, lane_polylines)
    
    path_waypoints = []
    for t in range(0, len(traj), 10):
        lane, point = find_map_waypoint(np.array((traj[t].center_x, traj[t].center_y, traj[t].heading)), lane_polylines)
        path_waypoints.append(lane_polylines[lane][point])

    after_waypoints = []
    for waypoint in lane_polylines[end_lane][end_point:]:
        after_waypoints.append(waypoint)
    if len(after_waypoints) < 100:
        if map_lanes[end_lane].exit_lanes:
            lane = map_lanes[end_lane].exit_lanes[0]
            for waypoint in lane_polylines[lane]:
                after_waypoints.append(waypoint)

    waypoints = np.concatenate([path_waypoints, after_waypoints[::5]], axis=0)

    # generate smooth route
    tx, ty, tyaw, tc, _ = generate_target_course(waypoints[:, 0], waypoints[:, 1])
    ref_line = np.column_stack([tx, ty, tyaw, tc])
    
    # get reference path at current timestep
    current_location = np.argmin(np.linalg.norm(ref_line[:, :2] - cur_pos[np.newaxis, :2], axis=-1))
    start_index = np.max([current_location, 0])
    ref_line = ref_line[start_index:start_index+1000]

    # add speed limit and traffic signal info to ref route
    line_info = np.zeros(shape=(ref_line.shape[0], 1))
    speed_limit = map_lanes[cur_lane].speed_limit_mph / 2.237
    ref_line = np.concatenate([ref_line, line_info], axis=-1)
    signals = [Point([signal.stop_point.x, signal.stop_point.y]) for signal in map_signals[timestep].lane_states if signal.state in [1, 4, 7]]

    for i in range(ref_line.shape[0]):
        if any([Point(ref_line[i, :2]).distance(signal) < 0.2 for signal in signals]):
            ref_line[i, 4] = 0 # red light
        else:
            ref_line[i, 4] = speed_limit

    return ref_line


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
        return np.column_stack((line_rotate, heading))


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


def ref_line_norm(ref_line, center, angle):
    xy = LineString(ref_line[:, 0:2])
    xy = affine_transform(xy, [1, 0, 0, 1, -center[0], -center[1]])
    xy = rotate(xy, -angle, origin=(0, 0), use_radians=True)
    yaw = wrap_to_pi(ref_line[:, 2] - angle)
    c = ref_line[:, 3]
    info = ref_line[:, 4]

    return np.column_stack((xy.coords, yaw, c, info))
