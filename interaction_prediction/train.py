import torch
import sys
sys.path.append('..')
import csv
import argparse
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import time
from model.GameFormer import GameFormer
from utils.inter_pred_utils import *


# define model training epoch
def training_epoch(train_data, model, optimizer, epoch):
    epoch_loss = []
    model.train()
    current = 0
    start_time = time.time()
    size = len(train_data)
    ADE,FDE = [],[]
    ADEp,FDEp = [],[]

    for batch in train_data:
        # prepare data
        inputs = {
            'ego_state': batch[0].to(args.local_rank),
            'neighbors_state': batch[1].to(args.local_rank),
            'map_lanes': batch[2].to(args.local_rank),
            'map_crosswalks': batch[3].to(args.local_rank),
        }

        ego_future = batch[4].to(args.local_rank)
        neighbor_future = batch[5].to(args.local_rank)

        # zero gradients for every batch
        optimizer.zero_grad()
        # query the model
        outputs = model(inputs)
        loss,future = level_k_loss(outputs, ego_future, neighbor_future, args.level)
        # back-propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        p_ade,p_fde,pr_ade,pr_fde = motion_metrics(future[0], ego_future, neighbor_future)
        ADE.append(p_ade)
        FDE.append(p_fde)
        ADEp.append(pr_ade)
        FDEp.append(pr_fde)

        # compute metrics
        current += args.batch_size
        epoch_loss.append(loss.item())

        if dist.get_rank() == 0:
            logging.info(
                f"\rTrain Progress: [{current:>6d}/{size*args.batch_size:>6d}]"+
                f"|Loss: {np.mean(epoch_loss):>.4f}|"+
                f"Pred-1:ADE{np.mean(ADE):>.4f}-FDE{np.mean(FDE):>.4f}|"+
                f"Pred-2:ADE{np.mean(ADEp):>.4f}-FDE{np.mean(FDEp):>.4f}|"+
                f"{(time.time()-start_time)/current:>.4f}s/sample"
                )
    
    return epoch_loss

# define model validation epoch
def validation_epoch(valid_data, model, epoch):
    epoch_metrics = MotionMetrics()

    model.eval()
    current = 0
    start_time = time.time()
    size = len(valid_data)
    epoch_loss = []
    ADE,FDE = [],[]
    ADEp,FDEp = [],[]

    logging.info(f'Validation...Epoch{epoch+1}')

    for batch in valid_data:
        # prepare data
        inputs = {
            'ego_state': batch[0].to(args.local_rank),
            'neighbors_state': batch[1].to(args.local_rank),
            'map_lanes': batch[2].to(args.local_rank),
            'map_crosswalks': batch[3].to(args.local_rank),
        }

        ego_future = batch[4].to(args.local_rank)
        neighbor_future = batch[5].to(args.local_rank)

        # query the model
        with torch.no_grad():
            outputs = model(inputs)
            loss,future = level_k_loss(outputs, ego_future, neighbor_future, args.level)

        # compute metrics
        epoch_loss.append(loss.item())
        egos = outputs[f'level_{args.level}_interactions'][:, :, :, :, :2]
        scores = outputs[f'level_{args.level}_scores']
     
        object_type = batch[6]
        ego = inputs['ego_state']
        actors = torch.stack([ego,inputs['neighbors_state'][:, 0]],dim=1)
        actors_future = torch.stack([ego_future, neighbor_future],dim=1)
        ego_ground_truth = torch.cat([actors[:, :, :, :5], actors_future], dim=2)
        ego_ground_truth = torch.cat([
            ego_ground_truth[:, :, :, :2], 
            actors[:,:, -1, 5:7].unsqueeze(2).expand(-1,-1, ego_ground_truth.shape[2], -1), 
            ego_ground_truth[:, :, :, 2:]
            ], dim=-1)
        
        egos = egos.permute(0,2,1,3,4)
        scores = scores.sum(1)
        scores = F.softmax(scores,dim=-1)

        p_ade,p_fde,pr_ade,pr_fde = motion_metrics(future[0], ego_future, neighbor_future)
        ADE.append(p_ade)
        FDE.append(p_fde)
        ADEp.append(pr_ade)
        FDEp.append(pr_fde)

        epoch_metrics.update_state(
                    egos, scores, 
                    ego_ground_truth, torch.ne(ego_ground_truth, 0).bool(), 
                    object_type.long()
                    )

        current += args.batch_size
        if dist.get_rank() == 0:
            logging.info(
                f"\rTrain Progress: [{current:>6d}/{size*args.batch_size:>6d}]"+
                f"|Loss: {np.mean(epoch_loss):>.4f}|"+
                f"Pred-1:ADE{np.mean(ADE):>.4f}-FDE{np.mean(FDE):>.4f}|"+
                f"Pred-2:ADE{np.mean(ADEp):>.4f}-FDE{np.mean(FDEp):>.4f}|"+
                f"{(time.time()-start_time)/current:>.4f}s/sample"
                )
        
    epoch_metrics = epoch_metrics.result()
    
    return epoch_metrics, epoch_loss

# Define model training process
def main():

    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.local_rank))

    set_seed(args.seed)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    model = GameFormer(
                modalities=args.modalities,
                encoder_layers=args.encoder_layers,
                decoder_levels=args.level,
                future_len=args.future_len, 
                neighbors_to_predict=args.neighbors_to_predict
                )

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
                                            optimizer, 
                                            milestones=[20, 22, 24, 26, 28], 
                                            gamma=0.5,
                                            verbose=True)
    
    # load ckpts:
    curr_ep = 0
    if args.load_dir != '':
        model_path = log_path + args.load_dir
        model_ckpts = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_ckpts['model_states'])
        optimizer.load_state_dict(model_ckpts['optim_states'])
        curr_ep = model_ckpts['current_ep']
        scheduler.step(curr_ep)
    
    # datasets:
    train_dataset = DrivingData(args.train_set+'/*')
    valid_dataset = DrivingData(args.valid_set+'/*')

    training_size = len(train_dataset)
    valid_size = len(valid_dataset)

    if dist.get_rank() == 0:
        logging.info(f'Length train: {training_size}; Valid: {valid_size}')

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_data = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        sampler=train_sampler, num_workers=args.workers
        )
    valid_data = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers
        )

    #start training:
    epochs = args.training_epochs

    for epoch in range(epochs):
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}")
        
        if epoch<=curr_ep and epoch!=0:
            continue

        train_data.sampler.set_epoch(epoch)
        valid_data.sampler.set_epoch(epoch)

        train_loss = training_epoch(train_data, model, optimizer, epoch)
        valid_metrics, val_loss = validation_epoch(valid_data, model, epoch)

        # save to training log
        log = {
            'epoch': epoch+1, 
            'train_loss': np.mean(train_loss), 'val_loss': np.mean(val_loss),
            'lr': optimizer.param_groups[0]['lr']
            }

        log.update(valid_metrics)

        if dist.get_rank() == 0:
            # log & save
            if epoch == 0:
                with open(log_path + f'train_log.csv', 'w') as csv_file: 
                    writer = csv.writer(csv_file) 
                    writer.writerow(log.keys())
                    writer.writerow(log.values())
            else:
                with open(log_path + f'train_log.csv', 'a') as csv_file: 
                    writer = csv.writer(csv_file)
                    writer.writerow(log.values())
            
            save_state = {
                'optim_states' : optimizer.state_dict(),
                'model_states' :model.state_dict(),
                'current_ep': epoch
            }
            torch.save(save_state, log_path + f'epochs_{epoch}.pth')

        # adjust learning rate
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interaction Prediction Training')
    parser.add_argument("--local_rank", type=int)
    # training
    parser.add_argument("--batch_size", type=int, help='training batch sizes', default=16)
    parser.add_argument("--training_epochs", type=int, help='training epochs', default=30)
    parser.add_argument("--learning_rate", type=float, help='training learning rates', default=1e-4)
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    # data & loggings
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1_IP")
    parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='')
    parser.add_argument('--train_set', type=str, help='path to train data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument("--workers", type=int, default=8, help="number of workers used for dataloader")
    # model
    parser.add_argument("--level", type=int, help='decoder reasoning levels (K)', default=3)
    parser.add_argument("--neighbors_to_predict", type=int, help='neighbors to predict, 1 for Waymo Joint Prediction', default=1)
    parser.add_argument("--modalities", type=int, help='joint num of modalities', default=6)
    parser.add_argument("--future_len", type=int, help='prediction horizons', default=80)
    parser.add_argument("--encoder_layers", type=int, help='encoder layers', default=6)
    args = parser.parse_args()

    main()