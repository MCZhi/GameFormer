import sys
sys.path.append("..")
import glob
import torch
import argparse
import logging
import pandas as pd
import numpy as np
from model.GameFormer import GameFormer
from data_process import *
from utils.open_loop_test_utils import *


class OpenLoopTestProcessor(DataProcess):
    def __init__(self):
        self.num_neighbors = 10
        self.hist_len = 11
        self.future_len = 50
        self.n_lanes = 4
        self.n_crosswalks = 2
        self.n_refline_waypoints = 1000

    def process_frame(self, timestep, sdc_id, tracks):      
        ego = self.ego_process(sdc_id, timestep, tracks) 
        neighbors, neighbors_to_predict = self.neighbors_process(sdc_id, timestep, tracks)
        agent_map_lanes = np.zeros(shape=(1+self.num_neighbors, self.n_lanes, 100, 16), dtype=np.float32)
        agent_map_crosswalk = np.zeros(shape=(1+self.num_neighbors, self.n_crosswalks, 100, 3), dtype=np.float32)

        agent_map_lanes[0], agent_map_crosswalk[0] = self.map_process(ego, timestep, type=1)
        for i in range(self.num_neighbors):
            if neighbors[i, -1, 0] != 0:
                agent_map_lanes[i+1], agent_map_crosswalk[i+1] = self.map_process(neighbors[i], timestep)

        ref_line = self.route_process(sdc_id, timestep, self.current_xyh, tracks)

        if ref_line is None:
            return

        ground_truth = self.ground_truth_process(sdc_id, timestep, tracks)
        ego, neighbors, map_lanes, map_crosswalk, ref_line, ground_truth = \
                self.normalize_data(ego, neighbors, agent_map_lanes, agent_map_crosswalk, ref_line, ground_truth, viz=False)
        obs = {'ego_state': ego, 'neighbors_state':neighbors, 'map_lanes': map_lanes, 
               'map_crosswalks': map_crosswalk, 'ref_line': ref_line}       
        
        return obs, neighbors_to_predict, ground_truth


def open_loop_test():
    # logging
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')
    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use device: {}".format(args.device))

    # load testing scenarios from csv file
    test_scenarios = pd.read_csv('waymo_candid_list.csv')

    # test files
    files = glob.glob(args.test_set+'/*')
    test_files = []
    test_id = 0

    # check if the required files exist
    required_files = test_scenarios['File'].drop_duplicates().values
    for file in required_files:
        file = f"{args.test_set}/uncompressed_scenario_validation_{file}"
        test_files.append(file)
        if file not in files:
            logging.error(f"File {file} does not exist!")
            sys.exit()

    # data processor
    processor = OpenLoopTestProcessor()

    # cache results
    scenario_ids = []
    collisions = []
    miss_rates = []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE = [], []

    # load model
    gameformer = GameFormer(modalities=6, neighbors_to_predict=10, future_len=50).to(args.device)
    gameformer.load_state_dict(torch.load(args.model_path, map_location=args.device))
    gameformer.eval()

    # iterate thru test files
    for file in test_files:
        scenarios = tf.data.TFRecordDataset(file)

        # iterate thru scenarios
        for scenario in scenarios:
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())
            scenario_id = parsed_data.scenario_id
            if scenario_id not in test_scenarios['Scenario ID'].values:
                continue
            
            test_id += 1
            logging.info(f"{test_id}/{len(test_scenarios)} Testing scenario: {scenario_id}")
            sdc_id = parsed_data.sdc_track_index
            timesteps = parsed_data.timestamps_seconds

            # build map
            processor.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

            # iterate thru timesteps
            for curr_t in range(10, len(timesteps)-50, 5):
                logging.info(f"Testing timestep: {curr_t}")
                scenario_ids.append(f'{scenario_id}_{curr_t}')
                data = processor.process_frame(curr_t, sdc_id, parsed_data.tracks)
                if data is None:
                    continue
                else:
                    obs, neighbor_ids, gt_future = data

                # prepare data
                inputs = {
                    'ego_state': torch.from_numpy(obs['ego_state']).unsqueeze(0).to(args.device),
                    'neighbors_state': torch.from_numpy(obs['neighbors_state']).unsqueeze(0).to(args.device),
                    'map_lanes': torch.from_numpy(obs['map_lanes']).unsqueeze(0).to(args.device),
                    'map_crosswalks': torch.from_numpy(obs['map_crosswalks']).unsqueeze(0).to(args.device),
                    'ref_line': torch.from_numpy(obs['ref_line']).unsqueeze(0).to(args.device)
                }

                ego_future = gt_future[0]
                neighbors_future = gt_future[1:]

                # level-k reasoning
                with torch.no_grad():
                    level_k_outputs = gameformer(inputs)

                level = len(level_k_outputs.keys()) // 2 - 1
                trajectories = level_k_outputs[f'level_{level}_interactions']
                scores = level_k_outputs[f'level_{level}_scores']
                trajectories = select_future(trajectories, scores)
                plan = trajectories[0].cpu()
                predictions = trajectories[1:].cpu().numpy()
 
                # compute metrics
                current_state = inputs['ego_state'][0, -1].cpu()
                xy = torch.cat([current_state[None, :2], plan])
                dxy = torch.diff(xy, dim=0)
                theta = torch.atan2(dxy[:, 1], dxy[:, 0].clip(min=1e-3)).unsqueeze(-1)
                plan = torch.cat([plan, theta], dim=-1).numpy()
                collision = check_collision(plan, neighbors_future, obs['ego_state'][-1, 5:], 
                                            obs['neighbors_state'][:, -1, 5:])
                collisions.append(collision)
                miss = check_ego_miss(plan, ego_future)
                miss_rates.append(miss)
                logging.info(f"Ego Collision: {collision}, Miss: {miss}")

                similarity = check_ego_similarity(plan, ego_future)
                similarity_1s.append(np.mean(similarity))
                similarity_3s.append(similarity[29])
                similarity_5s.append(similarity[49])
                logging.info(f"Plan Similarity@1s: {similarity[9]}, Similarity@3s: {similarity[29]}, Similarity@5s: {similarity[49]}")

                prediction_error = check_agent_prediction(predictions, neighbors_future)
                prediction_ADE.append(prediction_error[0])
                prediction_FDE.append(prediction_error[1])
                logging.info(f"Prediction ADE: {prediction_error[0]}, FDE: {prediction_error[1]}")
                logging.info(f"--------------------------------------------------")

                # plot scenario
                if args.render:
                    trajectories = transform_to_global_frame(curr_t, trajectories.cpu().numpy(), 
                                                             processor.current_xyh, neighbor_ids, parsed_data.tracks)
                    plot_scenario(curr_t, sdc_id, neighbor_ids, processor.map, processor.current_xyh, 
                                  parsed_data.tracks, trajectories, scores, args.name, scenario_id, args.save)
            
    # save results
    df = pd.DataFrame(data={'scenarios': scenario_ids, 'collision': collisions, 'miss': miss_rates, 
                            'Prediction_ADE': prediction_ADE, 'Prediction_FDE': prediction_FDE,
                            'Human_L2_1s': similarity_1s, 'Human_L2_3s': similarity_3s, 'Human_L2_5s': similarity_5s})
    df.to_csv(f'./testing_log/{args.name}/testing_log.csv')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Open-loop Testing')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--test_set', type=str, help='path to testing datasets')
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--render', action="store_true", help='if render the scenario (default: False)', default=False)
    parser.add_argument('--save', action="store_true", help='if save the rendered images (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cpu)', default='cpu')
    args = parser.parse_args()

    open_loop_test()
