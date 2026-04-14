

import glob
from multiprocessing import Pool
import subprocess
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import affine_transform, rotate
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import womd_lidar_utils
import argparse
import os

import sys
sys.path.append('/content/GameFusion')
sys.path.append('/content/GameFusion/interaction_prediction')
from interaction_prediction.data_process import DataProcess
from utils.data_utils import *

class DataProcessv1(DataProcess):
    def __init__(self, root_dir, point_dir='', save_dir='', merger_save_path='',
     ignore_vectorized_data=False, ignore_lidar_bev=False):
        super().__init__()
        self.data_files = root_dir
        self.root_dir = root_dir
        self.point_dir = point_dir
        self.save_dir = save_dir
        self.merger_save_path = merger_save_path
        self.ignore_vectorized_data = ignore_vectorized_data
        self.ignore_lidar_bev = ignore_lidar_bev

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
            vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=300, endpoint=False, dtype=int)]
          
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
            
            if added_cross_walks >= 4:
                break

        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32)

    def merge_sensors_with_scenario(self, shard_dataset, shard_id, split_type):
      os.makedirs("/content/data/lidar_and_camera", exist_ok=True)
      os.makedirs(self.merger_save_path, exist_ok=True)
      output_path = f"{self.merger_save_path}/merged_{split_type}_shard-{shard_id}.tfrecord"

      # First pass: collect all scenario IDs
      scenario_ids = []
      for data in shard_dataset:
          scenario = scenario_pb2.Scenario()
          scenario.ParseFromString(data.numpy())
          scenario_ids.append(scenario.scenario_id)
      print(f"Found {len(scenario_ids)} scenarios in shard")

      # Download in batches of 2
      batch_size = 2
      for i in tqdm(range(0, len(scenario_ids), batch_size), desc="Downloading sensor data" ):
          batch = scenario_ids[i:i+batch_size]
          paths_file = "/content/data/paths.txt"
          with open(paths_file, "w") as f:
              for sid in batch:
                  f.write(f"gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/lidar_and_camera/{split_type}/{sid}.tfrecord\n")
          
          result = subprocess.run(
              ["bash", "-c", f"cat {paths_file} | gsutil -m cp -I /content/data/lidar_and_camera/"],
              capture_output=True, text=True
          )
          # print(f"Batch {i//batch_size + 1}: downloaded {len(batch)} files")
          if result.returncode != 0:
              print(f"Error: {result.stderr[-300:]}")

      downloaded = os.listdir("/content/data/lidar_and_camera/")
      print(f"Total downloaded: {len(downloaded)} / {len(scenario_ids)}")

      # Second pass: merge and write
      writer = tf.io.TFRecordWriter(output_path)
      merged = 0

      for data in tqdm(shard_dataset, desc="Merging"):
          scenario = scenario_pb2.Scenario()
          scenario.ParseFromString(data.numpy())
          scenario_id = scenario.scenario_id

          lidar_cam_path = f"/content/data/lidar_and_camera/{scenario_id}.tfrecord"
          if os.path.exists(lidar_cam_path):
              lidar_data = next(iter(tf.data.TFRecordDataset(lidar_cam_path)))
              lidar_cam = scenario_pb2.Scenario()
              lidar_cam.ParseFromString(lidar_data.numpy())
              scenario.compressed_frame_laser_data.MergeFrom(lidar_cam.compressed_frame_laser_data)
              scenario.frame_camera_tokens.MergeFrom(lidar_cam.frame_camera_tokens)
              merged += 1

          writer.write(scenario.SerializeToString())

      writer.close()
      print(f"Done! Merged: {merged} / {len(scenario_ids)}")
      
    
    
    
    def extract_lidar_bev(self, points, x_min, x_max, y_min, y_max, z_min, z_max, xy_res, z_res):

        grid_x = round((x_max - x_min) / xy_res)
        grid_y = round((y_max - y_min) / xy_res)
        grid_z = round((z_max - z_min) / z_res)

        voxel_grid = np.zeros((grid_z, grid_x, grid_y), dtype=np.float32)

        x_vals = points[:, 0]
        y_vals = points[:, 1]
        z_vals = points[:, 2]

        # Correct formula: (value - min) / resolution
        x_bins = ((x_vals - x_min) / xy_res).astype(int)
        y_bins = ((y_vals - y_min) / xy_res).astype(int)
        z_bins = ((z_vals - z_min) / z_res).astype(int)

        # Clamp to valid range
        mask = (
            (0 <= x_bins) & (x_bins < grid_x) &
            (0 <= y_bins) & (y_bins < grid_y) &
            (0 <= z_bins) & (z_bins < grid_z)
        )

        voxel_grid[z_bins[mask], x_bins[mask], y_bins[mask]] = 1.0

        return voxel_grid

    def normalize_lidar_points(self, pts, center, angle):
        """
        Normalize LiDAR points by translating to the center and rotating by the negative angle.
        """
        # Translate
        pts_norm = pts.copy()
        pts_norm[:, 0] -= center[0]
        pts_norm[:, 1] -= center[1]
        
        # Rotate by -angle (standard 2D rotation matrix)
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        x_rot = cos_a * pts_norm[:, 0] - sin_a * pts_norm[:, 1]
        y_rot = sin_a * pts_norm[:, 0] + cos_a * pts_norm[:, 1]
        
        pts_norm[:, 0] = x_rot
        pts_norm[:, 1] = y_rot
        # Z unchanged
        
        return pts_norm
    
    def get_lidar_point(self, scenario):
        all_frames = []
        for frame in scenario.compressed_frame_laser_data:
            pose = tf.constant(list(frame.pose.transform), dtype=tf.float64)
            pose = tf.reshape(pose, [4, 4])
            calibs = {c.name: c for c in frame.laser_calibrations}

            all_points = []
            for laser in frame.lasers:
                calib = calibs[laser.name]
                if laser.name == 1:
                    xyz_ri1, _, xyz_ri2, _ = womd_lidar_utils.extract_top_lidar_points(
                        laser, pose, calib)
                else:
                    xyz_ri1, _, xyz_ri2, _ = womd_lidar_utils.extract_side_lidar_points(
                        laser, calib)
                all_points.append(xyz_ri1.numpy())
                all_points.append(xyz_ri2.numpy())

            pts = np.concatenate(all_points, axis=0)
            
            # Save original Z 
            z_vehicle = pts[:, 2].copy()
            
            # Transform XY to global frame for alignment with trajectories
            pose_np = pose.numpy()
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.concatenate([pts, ones], axis=1)
            pts_global = (pose_np @ pts_h.T).T[:, :3]
            
            # Replace Z with vehicle-frame Z
            pts_global[:, 2] = z_vehicle
            
            all_frames.append(pts_global)
        
        return all_frames

    def process_data(self, viz=True, test=False):

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
                    if parsed_data.tracks[tracks_to_predict[0].track_index].object_type == 1:
                        self.sdc_ids_list = [([tracks_list[1], tracks_list[0]], 1)]
                    else:
                        self.sdc_ids_list = [(tracks_list, 1)]
                else:
                    self.interactive_process(tracks_list, interact_list, parsed_data.tracks)

                lidar_frames = self.get_lidar_point(parsed_data)

                for pairs in self.sdc_ids_list:
                    sdc_ids, interesting = pairs[0], pairs[1]
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
                    ego, neighbors, map_lanes, map_crosswalks, ground_truth, region_dict = self.normalize_data(ego, neighbors,map_lanes,map_crosswalks,ground_truth, viz=viz)

                    center, angle = np.array(self.current_xyzh[0][:2]), self.current_xyzh[0][3]

                    if len(lidar_frames) > 0 and not self.ignore_lidar_bev:
                        bev_frames = []
                        for lidar_pts in lidar_frames:
                            lidar_pts_norm = self.normalize_lidar_points(lidar_pts, center, angle)
                            bev = self.extract_lidar_bev(lidar_pts_norm, -75, 75, -75, 75, -2, 4, 0.5, 0.5)
                            bev_frames.append(bev)
                        lidar_bev = np.array(bev_frames, dtype=np.uint8)
                    else:
                        lidar_bev = np.zeros((11, 12, 300, 300), dtype=np.uint8)

                    camera_array = np.zeros((11, 8, 256), dtype=np.float32)
                    if len(parsed_data.frame_camera_tokens) > 0:
                        for frame_idx, frame in enumerate(parsed_data.frame_camera_tokens):
                            if frame_idx >= 11:
                                break
                            for cam_idx, cam in enumerate(frame.camera_tokens):
                                if cam_idx >= 8:
                                    break
                                tokens = list(cam.tokens)
                                camera_array[frame_idx, cam_idx, :len(tokens)] = tokens

                    if self.point_dir == '':
                        region_dict = {6: np.zeros((6, 2))}
                    # save data
                    inter = 'interest' if interesting == 1 else 'r'
                    if not self.ignore_vectorized_data:
                        os.makedirs(self.save_dir, exist_ok=True)
                        filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"
                        if test:
                            np.savez_compressed(filename, ego=np.array(ego), neighbors=np.array(neighbors),
                                                map_lanes=np.array(map_lanes),
                                                map_crosswalks=np.array(map_crosswalks),
                                                object_type=np.array(object_type), region_6=np.array(region_dict[6]),
                                                object_index=np.array(object_index),
                                                current_state=np.array(self.current_xyzh[0]),
                                                lidar_bev=lidar_bev,
                                                camera_tokens=camera_array
                                                )
                        else:
                            np.savez_compressed(filename, ego=np.array(ego), neighbors=np.array(neighbors),
                                                map_lanes=np.array(map_lanes),
                                                map_crosswalks=np.array(map_crosswalks),
                                                object_type=np.array(object_type), region_6=np.array(region_dict[6]),
                                                object_index=np.array(object_index),
                                                current_state=np.array(self.current_xyzh[0]),
                                                gt_future_states=np.array(ground_truth),
                                                lidar_bev=lidar_bev,
                                                camera_tokens=camera_array
                                                )
                    # if not self.ignore_lidar_bev:
                    #     os.makedirs(self.save_dir + "/lidar_bev", exist_ok=True)
                    #     lidar_filename = self.save_dir + "/lidar_bev" + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"
                    #     np.savez_compressed(lidar_filename, lidar_bev=lidar_bev)

                self.pbar.update(1)

            self.pbar.close()
        print("bloom interaction prediction data processing done!")
    
def parallel_process(root_dir):
    print(root_dir)
    processor = DataProcessv1(root_dir=[root_dir], point_dir=point_path, save_dir=save_path, merger_save_path=merger_save_path, ignore_vectorized_data=ignore_vectorized_data, ignore_lidar_bev=ignore_lidar_bev)
    if merge_sensors and not load_all_shards:
        merge_sensors_with_scenario_wrapper(processor, shards_path, args.split_type)
    if process_data:
        processor.process_data(viz=debug,test=test)
    print(f'{root_dir}-done!')
    
def merge_sensors_with_scenario_wrapper(processor, shards_path, split_type):
    print("\nMerging sensors with scenario...")
    if split_type == 'testing':
        split_type = "validation"
        shard_id = "00001-of-00150"
    elif split_type == 'validation':
        shard_id = "00000-of-00150"
    else:
        split_type = 'training'
        shard_id = "00000-of-01000"
    cmd = [
            "gsutil", "-m", "cp",
            f"gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/{split_type}/{split_type}.tfrecord-{shard_id}",
            shards_path
        ]
    subprocess.run(cmd, check=True)
    print(f"{split_type}.tfrecord-{shard_id} shard downloaded successfully.\n")
    filenames = tf.io.matching_files(os.path.join(shards_path, f'{split_type}.tfrecord-*'))        
    train_dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    
    processor.merge_sensors_with_scenario(train_dataset, shard_id, split_type)
  
def main():
    parser = argparse.ArgumentParser(description='Data Processing Interaction Predictions')
    parser.add_argument('--shards_path', type=str, help='path to save processed data')
    parser.add_argument('--merger_save_path', type=str, help='path to save processed data')
    parser.add_argument('--load_all_shards',action='store_true',help='whether to load all shards')
    parser.add_argument('--merge_sensors',action='store_true',help='whether to merge sensors with scenario')
    parser.add_argument('--process_data',action='store_true',help='whether to process data')
    parser.add_argument('--load_path', type=str, help='path to dataset files')
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--point_path', type=str, help='path to load K-Means Anchors (Currently not included in the pipeline)', default='')
    parser.add_argument('--processes', type=int, help='multiprocessing process num', default=8)
    parser.add_argument('--debug', action="store_true", help='visualize processed data', default=False)
    parser.add_argument('--test', action="store_true", help='whether to process testing set', default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", help='use multiprocessing', default=False)
    parser.add_argument('--ignore_vectorized_data', action="store_true", help='ignore vector data', default=False)
    parser.add_argument('--ignore_lidar_bev', action="store_true", help='ignore lidar bev', default=False)
    parser.add_argument('--split_type', type=str, help='type of split to use', default='training')
    args = parser.parse_args()
        
    data_files = glob.glob(args.load_path+'/*')
    print(f"Found {len(data_files)} files")
    print(data_files[:5])  # see what it found
    save_path = args.save_path
    point_path = args.point_path
    debug = args.debug
    test = args.test
    merger_save_path = args.merger_save_path
    load_all_shards = args.load_all_shards
    process_data = args.process_data
    shards_path = args.shards_path
    merge_sensors = args.merge_sensors
    ignore_vectorized_data = args.ignore_vectorized_data
    ignore_lidar_bev = args.ignore_lidar_bev
    split_type = args.split_type


    os.makedirs(save_path, exist_ok=True)

    if args.use_multiprocessing:
        with Pool(processes=args.processes) as p:
            p.map(parallel_process, data_files)
    else:
        processor = DataProcessv1(root_dir=data_files, point_dir=point_path, save_dir=save_path, merger_save_path=merger_save_path, ignore_vectorized_data=ignore_vectorized_data, ignore_lidar_bev=ignore_lidar_bev)
        if merge_sensors and not load_all_shards:
            merge_sensors_with_scenario_wrapper(processor, shards_path, split_type)
        if process_data:
            processor.process_data(viz=debug,test=test)
    print('Done!')
    
  
if __name__ == "__main__":
    main()