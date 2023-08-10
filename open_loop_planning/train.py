import os
import sys
sys.path.append("..")
import csv
import torch
import argparse
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from model.GameFormer import GameFormer
from torch.utils.data import DataLoader
from utils.open_loop_train_utils import *


def train_epoch(data_loader, model, optimizer):
    epoch_loss = []
    epoch_metrics = []
    model.train()

    with tqdm(data_loader, desc="Training", unit="batch") as epoch:
        for batch in epoch:
            # prepare data
            inputs = {
                'ego_state': batch[0].to(args.device),
                'neighbors_state': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'ref_line': batch[4].to(args.device)
            }

            ego_future = batch[5][:, 0].to(args.device)
            neighbors_future = batch[5][:, 1:].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # call the mdoel
            optimizer.zero_grad()
            level_k_outputs = model(inputs)
            loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
            plan = results[:, 0]
            prediction = results[:, 1:]

            # loss backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # compute metrics
            metrics = motion_metrics(plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f"plannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, " +
                 f"predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}\n")
        
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, model):
    epoch_loss = []
    epoch_metrics = []
    model.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as epoch:
        for batch in epoch:
            # prepare data
            inputs = {
                'ego_state': batch[0].to(args.device),
                'neighbors_state': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'ref_line': batch[4].to(args.device)
            }

            ego_future = batch[5][:, 0].to(args.device)
            neighbors_future = batch[5][:, 1:].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # predict
            with torch.no_grad():
                level_k_outputs = model(inputs)
                loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
                plan = results[:, 0]
                prediction = results[:, 1:]

            # compute metrics
            metrics = motion_metrics(plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f"val-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, " +
                 f"val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics


def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up model
    gameformer = GameFormer(modalities=6, neighbors_to_predict=10, future_len=50, decoder_levels=args.levels)
    gameformer = gameformer.to(args.device)

    # set up optimizer
    optimizer = optim.AdamW(gameformer.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(args.train_set+'/*')
    valid_set = DrivingData(args.valid_set+'/*')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_metrics = train_epoch(train_loader, gameformer, optimizer)
        val_loss, val_metrics = valid_epoch(valid_loader, gameformer)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-plannerADE': train_metrics[0], 'train-plannerFDE': train_metrics[1], 
               'train-predictorADE': train_metrics[2], 'train-predictorFDE': train_metrics[3],
               'val-plannerADE': val_metrics[0], 'val-plannerFDE': val_metrics[1], 
               'val-predictorADE': val_metrics[2], 'val-predictorFDE': val_metrics[3]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(gameformer.state_dict(), f'training_log/{args.name}/predictor_{epoch+1}_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--levels', type=int, help='levels of reasoning', default=4)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    args = parser.parse_args()

    # Run
    model_training()
