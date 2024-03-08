import glob
import sys
sys.path.append("..")
import argparse
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from waymo_open_dataset.protos import scenario_pb2
from utils.data_utils import *
import os
import pickle

tf.config.set_visible_devices([], 'GPU')

class DataProcess(object):
    def __init__(
                self, 
                root_dir=[''],
                point_dir='',
                save_dir='',
                num_neighbors=32
                ):
        # parameters
        self.num_neighbors = num_neighbors
        self.hist_len = 11
        self.future_len = 80
        self.data_files = root_dir
        self.point_dir = point_dir
        self.save_dir = save_dir
    
    
    def build_points(self):
        self.points_dict = {}
        for obj_type in ['vehicle','pedestrian','cyclist']:
            for c in [6,32,64]:
                with open(self.point_dir + f'{obj_type}_{c}.pkl','rb') as reader:
                    data = pickle.load(reader)
                assert data.shape[0]==c
                self.points_dict[f'{obj_type}_{c}'] = data

    def build_map(self, map_features, dynamic_map_states):
        self.lanes = {}
        self.roads = {}
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
            elif map_type == 'road_line' or map_type == 'road_edge':
                self.roads[map_id] = map
            elif map_type == 'stop_sign':
                self.stop_signs[map_id] = map
            elif map_type == 'crosswalk': 
                self.crosswalks[map_id] = map
            elif map_type == 'speed_bump':
                self.speed_bumps[map_id] = map
            else:
                continue
                # raise TypeError

        # dynamic map features
        self.traffic_signals = dynamic_map_states

    def map_process(self, traj):
        '''
        Map point attributes
        self_point (x, y, h), left_boundary_point (x, y, h), right_boundary_pont (x, y, h), speed limit (float),
        self_type (int), left_boundary_type (int), right_boundary_type (int), traffic light (int), stop_point (bool), interpolating (bool), stop_sign (bool)
        '''
        vectorized_map = np.zeros(shape=(6, 300, 17))
        vectorized_crosswalks = np.zeros(shape=(4, 100, 3))
        agent_type = int(traj[-1][-1])

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
            candidate = depth_first_search(curr_lane, self.lanes, dist=lane_polylines[curr_lane][start:].shape[0], threshold=300)
            ref_lanes.extend(candidate)
        
        if agent_type != 2:
            # find current lanes' left and right lanes
            neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self.lanes, lane_polylines)

            # get neighbor lane's forward lanes
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self.lanes, dist=lane_polylines[neighbor_lane][start:].shape[0], threshold=300)
                ref_lanes.extend(candidate)
            
            # update reference lane ids
            ref_lane_ids.update(neighbor_lane_ids)

        # remove overlapping lanes
        ref_lanes = remove_overlapping_lane_seq(ref_lanes)
        
        # get traffic light controlled lanes and stop sign controlled lanes
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self.traffic_signals[self.hist_len-1].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        
        # add lanes to the array
        added_lanes = 0
        for i, s_lane in enumerate(ref_lanes):
            added_points = 0
            if i > 5:
                break
            
            # create a data cache
            cache_lane = np.zeros(shape=(500, 17))

            for lane in s_lane:
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]

                if added_points >= 500:
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
                    cache_lane[added_points, 15] = self.lanes[lane].interpolating

                    # traffic_light
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                        if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 3:
                            cache_lane[added_points, 14] = True
             
                    # add stop sign
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 16] = True

                    # count
                    added_points += 1
                    curr_index += 1

                    if added_points >= 500:
                        break             

            # scale the lane
            vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=300, endpoint=False, dtype=np.int)]
          
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
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int)]

            if detection.intersects(polygon):
                vectorized_crosswalks[added_cross_walks, :polyline.shape[0]] = polyline
                added_cross_walks += 1
            
            if added_cross_walks >= 4:
                break

        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32)

    def ego_process(self, sdc_ids, tracks):
        ego_states = np.zeros(shape=(2, self.hist_len, 9))
        self.current_xyzh = []
        for s,sdc_id in enumerate(sdc_ids):
            sdc_states = tracks[sdc_id].states[:self.hist_len]
            ego_type = tracks[sdc_id].object_type
            self.ego_type = ego_type

            # get the sdc current state
            self.current_xyzh.append( (tracks[sdc_id].states[self.hist_len-1].center_x, tracks[sdc_id].states[self.hist_len-1].center_y, 
                                tracks[sdc_id].states[self.hist_len-1].center_z, tracks[sdc_id].states[self.hist_len-1].heading) )

            # add sdc states into the array
            for i, sdc_state in enumerate(sdc_states):
                if sdc_state.valid:
                    ego_state = np.array([sdc_state.center_x, sdc_state.center_y, sdc_state.heading, sdc_state.velocity_x, 
                                        sdc_state.velocity_y, sdc_state.length, sdc_state.width, sdc_state.height, 
                                        ego_type])
                    ego_states[s,i] = ego_state

        return ego_states.astype(np.float32)

    def neighbors_process(self, sdc_ids, tracks):
        neighbors_states = np.zeros(shape=(self.num_neighbors, self.hist_len, 9))
        neighbors = []
        self.neighbors_id = []
        self.neighbors_type = []

        # search for nearby agents
        for e in range(len(sdc_ids)):
            for i, track in enumerate(tracks):
                track_states = track.states[:self.hist_len]
                if i not in sdc_ids and track_states[-1].valid:
                    xy = np.stack([track_states[-1].center_x, track_states[-1].center_y], axis=-1)
                    neighbors.append((i, np.linalg.norm(xy - self.current_xyzh[e][:2]))) 

        # sort the agents by distance
        sorted_neighbors = sorted(neighbors, key=lambda item: item[1])

        # add neighbor agents into the array
        added_num = 0
        appended_ids = set()
        for neighbor in sorted_neighbors:
            neighbor_id = neighbor[0]
            if neighbor_id in appended_ids:
                continue
            appended_ids.add(neighbor_id)

            neighbor_states = tracks[neighbor_id].states[:self.hist_len]
            neighbor_type = tracks[neighbor_id].object_type
            self.neighbors_type.append(neighbor_type)
            if neighbor_type <= 0 or neighbor_type > 3:
                neighbor_type = 0
                
            self.neighbors_id.append(neighbor_id)
            
            for i, neighbor_state in enumerate(neighbor_states):
                if neighbor_state.valid: 
                    neighbors_states[added_num, i] = np.array([neighbor_state.center_x, neighbor_state.center_y, neighbor_state.heading,  neighbor_state.velocity_x, 
                                                               neighbor_state.velocity_y, neighbor_state.length, neighbor_state.width, neighbor_state.height, 
                                                               neighbor_type])
            added_num += 1

            # only consider 'num_neihgbors' agents
            if added_num >= self.num_neighbors:
                break

        return neighbors_states.astype(np.float32), self.neighbors_id

    def ground_truth_process(self, sdc_ids, tracks):
        ground_truth = np.zeros(shape=(2, self.future_len, 5))
        
        for j, sdc_id in enumerate(sdc_ids):
            track_states = tracks[sdc_id].states[self.hist_len:]
            for i, track_state in enumerate(track_states):
                ground_truth[j, i] = np.stack([track_state.center_x, track_state.center_y, track_state.heading, 
                                            track_state.velocity_x, track_state.velocity_y], axis=-1)

        return ground_truth.astype(np.float32)
    
    def get_static_region(self,ego):
        region_dict = {}
        for c in [6,32,64]:
            region = []
            for i,n_type in enumerate(self.object_type):
                if n_type==2:
                    obj_type = 'pedestrian'
                elif n_type==3:
                    obj_type = 'cyclist'
                else:
                    obj_type = 'vehicle'
                data = self.points_dict[f'{obj_type}_{c}']
                if i==0:
                    region.append(data)
                    continue
                x,y = data[:,0], data[:,1]
                p_x,p_y,theta = ego[i,-1,0],ego[i,-1,1],ego[i,-1,2]
                new_x = np.cos(-theta)*x + np.sin(-theta)*y + p_x
                new_y = -np.sin(-theta)*x + np.cos(-theta)*y + p_y
                region.append(np.stack([new_x,new_y],axis=1))
            region_dict[c] = np.array(region,dtype=np.float32)
        return region_dict

    def normalize_data(self, ego, neighbors, map_lanes, map_crosswalks, ground_truth, viz=False):
        # get the center and heading (local view)
        center, angle = ego[0].copy()[-1][:2], ego[0].copy()[-1][2]
        
        # normalize agent trajectories
        ego[0, :, :5] = agent_norm(ego[0], center, angle, impute=True)
        ego[1, :, :5] = agent_norm(ego[1], center, angle, impute=True)

        ground_truth[0] = agent_norm(ground_truth[0], center, angle) 
        ground_truth[1] = agent_norm(ground_truth[1], center, angle) 

        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i, :, :5] = agent_norm(neighbors[i], center, angle, impute=True) 

        if self.point_dir != '':
            region_dict = self.get_static_region(ego) 
        else:
            region_dict = None      

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

        # visulization
        if viz:
            for i in range(ego.shape[0]):
                rect = plt.Rectangle((ego[i,-1, 0]-ego[i,-1, 5]/2, ego[i,-1, 1]-ego[i,-1, 6]/2), ego[i,-1, 5], ego[i,-1, 6], linewidth=2, color='r', alpha=0.6, zorder=3,
                                    transform=mpl.transforms.Affine2D().rotate_around(*(ego[i,-1, 0], ego[i,-1, 1]), ego[i,-1, 2]) + plt.gca().transData)
                plt.gca().add_patch(rect)

                future = ground_truth[i][ground_truth[i][:, 0] != 0]
                plt.plot(future[:, 0], future[:, 1], 'r', linewidth=1, zorder=3)
            
            for i in range(neighbors.shape[0]):
                if neighbors[i, -1, 0] != 0:
                    rect = plt.Rectangle((neighbors[i, -1, 0]-neighbors[i, -1, 5]/2, neighbors[i, -1, 1]-neighbors[i, -1, 6]/2), 
                                          neighbors[i, -1, 5], neighbors[i, -1, 6], linewidth=1.5, color='m', alpha=0.6, zorder=3,
                                          transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]), neighbors[i, -1, 2]) + plt.gca().transData)
                    plt.gca().add_patch(rect)
            
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
                        plt.plot(centerline[:, 0], centerline[:, 1], 'k', linewidth=0.5) # plot centerline

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk = crosswalk[crosswalk[:, 0] != 0]
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=1) # plot crosswalk
            
            if self.point_dir != '':
                for i in range(region_dict[32].shape[0]):
                    plt.scatter(region_dict[32][i,:,0],region_dict[32][i,:,1],marker='*',s=10)

            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.close()

        return ego, neighbors, map_lanes, map_crosswalks, ground_truth,region_dict
    
    def interactive_process(self,tracks_list,interesting_ids,tracks):
        self.sdc_ids_list = []

        for ego_id in tracks_list:

            ego_state = tracks[ego_id].states[self.hist_len-1]
            ego_xy = np.array([ego_state.center_x, ego_state.center_y])

            candidate_tracks = []
            cnt = 2
            if len(tracks_list)==1:
                for i, track in enumerate(tracks):
                    track_states = track.states[self.hist_len-1]
                    if i != ego_id and track_states.valid:
                        tracks_xy = np.array([track_states.center_x, track_states.center_y])
                        candidate_tracks.append((i, np.linalg.norm(tracks_xy - ego_xy)))
            else:
                for t in tracks_list:
                    if t!=ego_id:
                        if t in interesting_ids and ego_id in interesting_ids:
                            self.sdc_ids_list.append(((ego_id, t), 1))
                            cnt -= 1
                            continue

                        track_states = tracks[t].states[self.hist_len-1]
                        tracks_xy = np.array([track_states.center_x, track_states.center_y])
                        candidate_tracks.append((t, np.linalg.norm(tracks_xy - ego_xy)))
            sorted_candidate = sorted(candidate_tracks, key=lambda item: item[1])[:cnt]

            for can in sorted_candidate:
                self.sdc_ids_list.append(((ego_id, can[0]), 0))

    def process_data(self, viz=True,test=False):
        
        if self.point_dir != '':
            self.build_points()

        for data_file in self.data_files:
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            for data in dataset:
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())
                
                scenario_id = parsed_data.scenario_id

                self.scenario_id = scenario_id
                objects_of_interest = parsed_data.objects_of_interest

                tracks_to_predict = parsed_data.tracks_to_predict
                id_list = {}
                tracks_list = []
                for ids in tracks_to_predict:
                    id_list[parsed_data.tracks[ids.track_index].id] = ids.track_index
                    tracks_list.append(ids.track_index)
                interact_list = []
                for int_id in objects_of_interest:
                    interact_list.append(id_list[int_id])

                self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

                if test:
                    if parsed_data.tracks[tracks_to_predict[0].track_index].object_type==1:
                        self.sdc_ids_list = [([tracks_list[1], tracks_list[0]],1)]
                    else:
                        self.sdc_ids_list = [(tracks_list,1)] 
                else:
                    self.interactive_process(tracks_list, interact_list, parsed_data.tracks)

                for pairs in self.sdc_ids_list:
                    sdc_ids, interesting = pairs[0], pairs[1]                   
                    # process data
                    ego = self.ego_process(sdc_ids, parsed_data.tracks)

                    ego_type = parsed_data.tracks[sdc_ids[0]].object_type
                    neighbor_type = parsed_data.tracks[sdc_ids[1]].object_type
                    object_type = np.array([ego_type, neighbor_type])
                    self.object_type = object_type
                    ego_index = parsed_data.tracks[sdc_ids[0]].id
                    neighbor_index = parsed_data.tracks[sdc_ids[1]].id
                    object_index = np.array([ego_index, neighbor_index])

                    neighbors, _ = self.neighbors_process(sdc_ids, parsed_data.tracks)
                    map_lanes = np.zeros(shape=(2, 6, 300, 17), dtype=np.float32)
                    map_crosswalks = np.zeros(shape=(2, 4, 100, 3), dtype=np.float32)
                    map_lanes[0], map_crosswalks[0] = self.map_process(ego[0])
                    map_lanes[1], map_crosswalks[1] = self.map_process(ego[1])

                    if test:
                        ground_truth = np.zeros((2, self.future_len, 5))
                    else:
                        ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
                    ego, neighbors, map_lanes, map_crosswalks, ground_truth,region_dict = self.normalize_data(ego, neighbors, map_lanes, map_crosswalks, ground_truth, viz=viz)

                    if self.point_dir == '':
                        region_dict = {6:np.zeros((6,2))}
                    # save data
                    inter = 'interest' if interesting==1 else 'r'
                    filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"
                    if test:
                        np.savez(filename, ego=np.array(ego), neighbors=np.array(neighbors), map_lanes=np.array(map_lanes), 
                        map_crosswalks=np.array(map_crosswalks),object_type=np.array(object_type),region_6=np.array(region_dict[6]),
                        object_index=np.array(object_index),current_state=np.array(self.current_xyzh[0]))
                    else:
                        np.savez(filename, ego=np.array(ego), neighbors=np.array(neighbors), map_lanes=np.array(map_lanes), 
                        map_crosswalks=np.array(map_crosswalks),object_type=np.array(object_type),region_6=np.array(region_dict[6]),
                        object_index=np.array(object_index),current_state=np.array(self.current_xyzh[0]),gt_future_states=np.array(ground_truth))
                
                self.pbar.update(1)

            self.pbar.close()

def parallel_process(root_dir):
    print(root_dir)
    processor = DataProcess(root_dir=[root_dir], point_dir=point_path, save_dir=save_path) 
    processor.process_data(viz=debug,test=test)
    print(f'{root_dir}-done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing Interaction Predictions')
    parser.add_argument('--load_path', type=str, help='path to dataset files')
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--point_path', type=str, help='path to load K-Means Anchors (Currently not included in the pipeline)', default='')
    parser.add_argument('--processes', type=int, help='multiprocessing process num', default=8)
    parser.add_argument('--debug', action="store_true", help='visualize processed data', default=False)
    parser.add_argument('--test', action="store_true", help='whether to process testing set', default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", help='use multiprocessing', default=False)
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    save_path = args.save_path
    point_path = args.point_path
    debug = args.debug
    test = args.test
    os.makedirs(save_path, exist_ok=True)

    if args.use_multiprocessing:
        with Pool(processes=args.processes) as p:
            p.map(parallel_process, data_files)
    else:
        processor = DataProcess(root_dir=data_files, point_dir=point_path, save_dir=save_path) 
        processor.process_data(viz=debug,test=test)
    print('Done!')
  
