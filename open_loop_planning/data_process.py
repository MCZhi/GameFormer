import os
import sys
sys.path.append("..")
import glob
import random
import argparse
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from waymo_open_dataset.protos import scenario_pb2
from utils.data_utils import *

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')


# Data process
class DataProcess(object):
    def __init__(self, files):
        self.num_neighbors = 10
        self.hist_len = 11
        self.future_len = 50
        self.n_lanes = 6
        self.n_crosswalks = 4
        self.n_refline_waypoints = 1000
        self.data_files = files

    def build_map(self, map_features, dynamic_map_states):
        self.lanes = {}
        self.roads = {}
        self.road_edges = {}
        self.road_lines = {}
        self.stop_signs = {}
        self.crosswalks = {}
        self.speed_bumps = {}

        # static map features
        for map in map_features:
            map_type = map.WhichOneof("feature_data")
            map_id = map.id
            map = getattr(map, map_type)

            if map_type == 'lane':
                self.lanes[map_id] = map
            elif map_type == 'road_line':
                self.road_lines[map_id] = map
            elif map_type == 'road_edge':
                self.road_edges[map_id] = map
            elif map_type == 'stop_sign':
                self.stop_signs[map_id] = map
            elif map_type == 'crosswalk': 
                self.crosswalks[map_id] = map
            elif map_type == 'speed_bump':
                self.speed_bumps[map_id] = map
            else:
                raise TypeError

        self.roads.update(self.road_edges)
        self.roads.update(self.road_lines)

        # dynamic map features
        self.traffic_signals = dynamic_map_states

        # all map features
        self.map = {"lane": self.lanes, "road_line": self.road_lines, "road_edge": self.road_edges, "crosswalk": self.crosswalks, 
                    "speed_bump": self.speed_bumps, "stop_sign": self.stop_signs, "dynamic_map_states": self.traffic_signals}

    def map_process(self, traj, timestep, type=None):
        '''
        Map point attributes
        self_point (x, y, h), left_boundary_point (x, y, h), right_boundary_point (x, y, h), speed limit (float),
        self_type (int), left_boundary_type (int), right_boundary_type (int), 
        traffic light (int), interpolating (bool), stop_sign (bool)
        '''
        vectorized_map = np.zeros(shape=(self.n_lanes, 100, 16))
        vectorized_crosswalks = np.zeros(shape=(self.n_crosswalks, 100, 3))
        agent_type = int(traj[-1][-1]) if type is None else type

        # get all lane polylines
        lane_polylines = get_polylines(self.lanes)

        # get all road lines and edges polylines
        road_polylines = get_polylines(self.roads)

        # find current lanes for the agent
        ref_lane_ids = find_reference_lanes(agent_type, traj, lane_polylines)

        # find candidate lanes
        ref_lanes = []

        # get current lane's forward lanes
        for curr_lane, start in ref_lane_ids.items():
            candidate = depth_first_search(curr_lane, self.lanes, dist=lane_polylines[curr_lane][start:].shape[0], threshold=200)
            ref_lanes.extend(candidate)
        
        if agent_type != 2:
            # find current lanes' left and right lanes
            neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self.lanes, lane_polylines)

            # get neighbor lane's forward lanes
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self.lanes, dist=lane_polylines[neighbor_lane][start:].shape[0], 
                                               threshold=200)
                ref_lanes.extend(candidate)
            
            # update reference lane ids
            ref_lane_ids.update(neighbor_lane_ids)

        # remove overlapping lanes
        ref_lanes = remove_overlapping_lane_seq(ref_lanes)

        # get traffic light controlled lanes and stop sign controlled lanes
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self.traffic_signals[timestep].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        
        # add lanes to the array
        added_lanes = 0
        for i, s_lane in enumerate(ref_lanes):
            added_points = 0
            if i >= self.n_lanes:
                break
            
            # create a data cache
            cache_lane = np.zeros(shape=(200, 16))

            for lane in s_lane:
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]

                if added_points >= 200: # max 100 meters (200 road points)
                    break      

                # add info to the array
                for point in self_line:
                    # self_point and type
                    cache_lane[added_points, 0:3] = point
                    cache_lane[added_points, 10] = self.lanes[lane].type

                    # left_boundary_point and type
                    for left_boundary in self.lanes[lane].left_boundaries:
                        left_boundary_id = left_boundary.boundary_feature_id
                        left_start = left_boundary.lane_start_index
                        left_end = left_boundary.lane_end_index
                        left_boundary_type = left_boundary.boundary_type # road line type
                        if left_boundary_type == 0:
                            left_boundary_type = self.roads[left_boundary_id].type + 8 # road edge type
                        
                        if left_start <= curr_index <= left_end:
                            left_boundary_line = road_polylines[left_boundary_id]
                            nearest_point = find_neareast_point(point, left_boundary_line)
                            cache_lane[added_points, 3:6] = nearest_point
                            cache_lane[added_points, 11] = left_boundary_type

                    # right_boundary_point and type
                    for right_boundary in self.lanes[lane].right_boundaries:
                        right_boundary_id = right_boundary.boundary_feature_id
                        right_start = right_boundary.lane_start_index
                        right_end = right_boundary.lane_end_index
                        right_boundary_type = right_boundary.boundary_type # road line type
                        if right_boundary_type == 0:
                            right_boundary_type = self.roads[right_boundary_id].type + 8 # road edge type

                        if right_start <= curr_index <= right_end:
                            right_boundary_line = road_polylines[right_boundary_id]
                            nearest_point = find_neareast_point(point, right_boundary_line)
                            cache_lane[added_points, 6:9] = nearest_point
                            cache_lane[added_points, 12] = right_boundary_type

                    # speed limit
                    cache_lane[added_points, 9] = self.lanes[lane].speed_limit_mph / 2.237

                    # interpolating
                    cache_lane[added_points, 14] = self.lanes[lane].interpolating

                    # traffic_light
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]

                    # add stop sign
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 15] = True

                    # count
                    added_points += 1
                    curr_index += 1

                    if added_points >= 200:
                        break             

            # scale the lane
            vectorized_map[i] = cache_lane[::2]
          
            # count
            added_lanes += 1

        # find surrounding crosswalks and add them to the array
        added_cross_walks = 0
        detection = Polygon([(0, -5), (50, -20), (50, 20), (0, 5)])
        detection = affine_transform(detection, [1, 0, 0, 1, traj[-1][0], traj[-1][1]])
        detection = rotate(detection, traj[-1][2], origin=(traj[-1][0], traj[-1][1]), use_radians=True)

        for _, crosswalk in self.crosswalks.items():
            polygon = Polygon([(point.x, point.y) for point in crosswalk.polygon])
            polyline = polygon_completion(crosswalk.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=int)]

            if detection.intersects(polygon):
                vectorized_crosswalks[added_cross_walks, :polyline.shape[0]] = polyline
                added_cross_walks += 1
            
            if added_cross_walks >= self.n_crosswalks: 
                break

        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32)

    def ego_process(self, sdc_id, timestep, tracks):
        ego_states = np.zeros(shape=(self.hist_len, 9))
        ego_type = 0
        sdc_states = tracks[sdc_id].states[timestep+1-self.hist_len:timestep+1]
    
        # get the sdc current state
        self.current_xyh = np.array((tracks[sdc_id].states[timestep].center_x, tracks[sdc_id].states[timestep].center_y, 
                                     tracks[sdc_id].states[timestep].heading), dtype=np.float32)

        # add sdc states into the array
        for i, sdc_state in enumerate(sdc_states):
            ego_state = np.array([sdc_state.center_x, sdc_state.center_y, sdc_state.heading, sdc_state.velocity_x, 
                                  sdc_state.velocity_y, sdc_state.length, sdc_state.width, sdc_state.height, ego_type])
            ego_states[i] = ego_state

        return ego_states.astype(np.float32)

    def neighbors_process(self, sdc_id, timestep, tracks):
        neighbors_states = np.zeros(shape=(self.num_neighbors, self.hist_len, 9))
        neighbors = {}
        self.neighbors_id = []

        # search for nearby agents
        for i, track in enumerate(tracks):
            track_states = track.states[timestep+1-self.hist_len:timestep+1]
            if i != sdc_id and track_states[-1].valid:
                neighbors[i] = np.stack([track_states[-1].center_x, track_states[-1].center_y], axis=-1)

        # sort the agents by distance
        sorted_neighbors = sorted(neighbors.items(), key=lambda item: np.linalg.norm(item[1] - self.current_xyh[:2]))

        # add neighbor agents into the array
        added_num = 0
        for neighbor in sorted_neighbors:
            neighbor_id = neighbor[0]
            neighbor_states = tracks[neighbor_id].states[timestep+1-self.hist_len:timestep+1]
            neighbor_type = tracks[neighbor_id].object_type
            self.neighbors_id.append(neighbor_id)

            for i, neighbor_state in enumerate(neighbor_states):
                if neighbor_state.valid: 
                    neighbors_states[added_num, i] = np.array([neighbor_state.center_x, neighbor_state.center_y, 
                                                               neighbor_state.heading, neighbor_state.velocity_x, 
                                                               neighbor_state.velocity_y, neighbor_state.length, 
                                                               neighbor_state.width, neighbor_state.height, neighbor_type])
            added_num += 1

            # only consider 'num_neihgbors' agents
            if added_num >= self.num_neighbors:
                break

        return neighbors_states.astype(np.float32), self.neighbors_id

    def ground_truth_process(self, sdc_id, timestep, tracks):
        ground_truth = np.zeros(shape=(1+self.num_neighbors, self.future_len, 5))

        track_states = tracks[sdc_id].states[timestep+1:timestep+self.future_len+1]
        for i, track_state in enumerate(track_states):
            ground_truth[0, i] = np.stack([track_state.center_x, track_state.center_y, track_state.heading, 
                                           track_state.velocity_x,  track_state.velocity_y], axis=-1)

        for i, id in enumerate(self.neighbors_id):
            track_states = tracks[id].states[timestep+1:timestep+self.future_len+1]
            for j, track_state in enumerate(track_states):
                ground_truth[i+1, j] = np.stack([track_state.center_x, track_state.center_y, track_state.heading, 
                                                 track_state.velocity_x,  track_state.velocity_y], axis=-1)

        return ground_truth.astype(np.float32)
    
    def route_process(self, sdc_id, timestep, cur_pos, tracks):
        # find reference paths according to the gt trajectory
        gt_path = tracks[sdc_id].states

        # remove rare cases
        try:
            route = find_route(gt_path, timestep, cur_pos, self.lanes, self.traffic_signals)
        except:
            return None

        ref_path = np.array(route, dtype=np.float32)

        if ref_path.shape[0] < self.n_refline_waypoints:
            repeated_last_point = np.repeat(ref_path[np.newaxis, -1], self.n_refline_waypoints-ref_path.shape[0], axis=0)
            ref_path = np.append(ref_path, repeated_last_point, axis=0)

        return ref_path

    def normalize_data(self, ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth, viz=False):
        # get the center and heading (local view)
        center, angle = self.current_xyh[:2], self.current_xyh[2]

        # normalize agent trajectories
        ego[:, :5] = agent_norm(ego, center, angle)
        ground_truth[0] = agent_norm(ground_truth[0], center, angle) 

        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i, :, :5] = agent_norm(neighbors[i], center, angle, impute=True)
                ground_truth[i+1] = agent_norm(ground_truth[i+1], center, angle)            

        # normalize map points
        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            crosswalks = map_crosswalks[i]

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    lane[:, :9] = map_norm(lane, center, angle)

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk[:, :3] = map_norm(crosswalk, center, angle)

        # normalize ref line
        ref_line = ref_line_norm(ref_line, center, angle).astype(np.float32)

        # visulization
        if viz:
            rect = plt.Rectangle((ego[-1, 0]-ego[-1, 5]/2, ego[-1, 1]-ego[-1, 6]/2), ego[-1, 5], ego[-1, 6], 
                                 linewidth=2, color='r', alpha=0.6, zorder=3,
                                transform=mpl.transforms.Affine2D().rotate_around(*(ego[-1, 0], ego[-1, 1]), ego[-1, 2]) + plt.gca().transData)
            plt.gca().add_patch(rect)

            plt.plot(ref_line[:, 0], ref_line[:, 1], 'y', linewidth=2, zorder=4)

            future = ground_truth[0][ground_truth[0][:, 0] != 0]
            plt.plot(future[:, 0], future[:, 1], 'r', linewidth=3, zorder=3)

            for i in range(neighbors.shape[0]):
                if neighbors[i, -1, 0] != 0:
                    rect = plt.Rectangle((neighbors[i, -1, 0]-neighbors[i, -1, 5]/2, neighbors[i, -1, 1]-neighbors[i, -1, 6]/2), 
                                          neighbors[i, -1, 5], neighbors[i, -1, 6], linewidth=2, color='m', alpha=0.6, zorder=3,
                                          transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]), neighbors[i, -1, 2]) + plt.gca().transData)
                    plt.gca().add_patch(rect)
                    future = ground_truth[i+1][ground_truth[i+1][:, 0] != 0]
                    plt.plot(future[:, 0], future[:, 1], 'm', linewidth=3, zorder=3)

            for i in range(map_lanes.shape[0]):
                lanes = map_lanes[i]
                crosswalks = map_crosswalks[i]

                for j in range(map_lanes.shape[1]):
                    lane = lanes[j]
                    if lane[0][0] != 0:
                        centerline = lane[:, 0:2]
                        centerline = centerline[centerline[:, 0] != 0]
                        left = lane[:, 3:5]
                        left = left[left[:, 0] != 0]
                        right = lane[:, 6:8]
                        right = right[right[:, 0] != 0]
                        plt.plot(centerline[:, 0], centerline[:, 1], 'c', linewidth=3) # plot centerline
                        plt.plot(left[:, 0], left[:, 1], 'k', linewidth=3) # plot left boundary
                        plt.plot(right[:, 0], right[:, 1], 'k', linewidth=3) # plot left boundary

                for k in range(map_crosswalks.shape[1]):
                    crosswalk = crosswalks[k]
                    if crosswalk[0][0] != 0:
                        crosswalk = crosswalk[crosswalk[:, 0] != 0]
                        plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4) # plot crosswalk

            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        return ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth

    def process_data(self, save_path, viz=True):
        for data_file in self.data_files:
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            for data in dataset:
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())
                
                scenario_id = parsed_data.scenario_id
                sdc_id = parsed_data.sdc_track_index
                time_len = len(parsed_data.tracks[sdc_id].states)
                self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

                for timestep in range(self.hist_len-1, time_len-self.future_len, 5):
                    # process data
                    ego = self.ego_process(sdc_id, timestep, parsed_data.tracks)
                    ref_line = self.route_process(sdc_id, timestep, self.current_xyh, parsed_data.tracks)
                    if ref_line is None:
                        continue

                    neighbors, _ = self.neighbors_process(sdc_id, timestep, parsed_data.tracks)
                    map_lanes = np.zeros(shape=(self.num_neighbors+1, self.n_lanes, 100, 16), dtype=np.float32)
                    map_crosswalks = np.zeros(shape=(self.num_neighbors+1, self.n_crosswalks, 100, 3), dtype=np.float32)
                    map_lanes[0], map_crosswalks[0] = self.map_process(ego, timestep, type=1)

                    for i in range(self.num_neighbors):
                        if neighbors[i, -1, 0] != 0:
                            map_lanes[i+1], map_crosswalks[i+1] = self.map_process(neighbors[i], timestep)

                    ground_truth = self.ground_truth_process(sdc_id, timestep, parsed_data.tracks)
                    ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth = \
                        self.normalize_data(ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth, viz=viz)

                    # save data
                    filename = f"{save_path}/{scenario_id}_{timestep}.npz"
                    np.savez(filename, ego=ego, neighbors=neighbors, map_lanes=map_lanes, map_crosswalks=map_crosswalks, 
                             ref_line=ref_line, gt_future_states=ground_truth)
                
                self.pbar.update(1)

            self.pbar.close()


def multiprocessing(data_files):
    processor = DataProcess([data_files]) 
    processor.process_data(save_path, viz=False)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--load_path', type=str, help='path to dataset files')
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--debug', action="store_true", help='visualize processed data', default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", help='use multiprocessing', default=False)
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    if args.use_multiprocessing:
        with Pool() as p:
            p.map(multiprocessing, data_files)
    else:
        processor = DataProcess(data_files) 
        processor.process_data(save_path, viz=args.debug)
        print('Done!')
