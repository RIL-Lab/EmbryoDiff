import os
import torch
import pickle
import csv
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.MFHE.video_dataset import collate_fn as collate_fn_MFHE
from data.SFHE.video_dataset import collate_fn as collate_fn_SFHE
from tqdm import tqdm
from utils import func_eval_modified
from collections import defaultdict


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, logger, event_list, config, device):

        """
        Init a traniner for training and model testing.
        """ 
        self.config = config
        self.num_classes = len(event_list)
        self.event_list = event_list
        self.set_sampling_seed = config["train"]["set_sampling_seed"]
        self.device = device
        
        self.logger = logger

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.model = model
        self.logger.info(f"Model Size:{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def get_train_loader(self):
        batch_size = self.config["train"]["batch_size"]
        collate_fn = collate_fn_MFHE if self.config["data"]["dataset_name"] == "MFHE" else collate_fn_SFHE
        num_workers = self.config["data"]["num_workers"]

        train_train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        
        return train_train_loader
    
    def get_test_loader(self):
        batch_size = self.config["test"]["batch_size"]
        collate_fn = collate_fn_MFHE if self.config["data"]["dataset_name"] == "MFHE" else collate_fn_SFHE
        num_workers = self.config["data"]["num_workers"]

        test_test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        
        return test_test_loader
    
    def configure_optimizer(self):
        learning_rate = self.config["train"].get("learning_rate", 1e-4)
        weight_decay = self.config["train"].get("weight_decay", 1e-2)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        return optimizer
    
    def configure_schedular(self, optimizer):
        from torch.optim.lr_scheduler import CosineAnnealingLR

        schedular = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.config["train"].get("num_epochs", 200),
            eta_min=1e-7,
            last_epoch=-1
        )

        return schedular
    
    def init_test_logit_dict(self):
        self.test_logits_dict = defaultdict(list)

    def train(self):
        device = self.device
        self.model.to(device)

        optimizer = self.configure_optimizer()
        schedular = self.configure_schedular(optimizer)
        optimizer.zero_grad()
        
        # config training loss functions
        if self.config["train"]["class_weighting"]:
            class_weights = self.train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean') 

        bce_criterion = nn.CrossEntropyLoss(reduction="mean") # for boundary loss
        mse_criterion = nn.MSELoss(reduction='none') # for temporal smoothing loss
        
        # build dataloader
        train_train_loader =self.get_train_loader()
        test_test_loader = self.get_test_loader()
        
        # create tensorboard writter
        result_dir = self.config["train"].get("result_dir", "./")
        tb_logger = None
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            tb_logger = SummaryWriter(result_dir)

        ############## training ################
        restore_epoch = -1
        step = 1
        self.best_en_acc = 0
        self.best_de_acc = 0
        self.best_encoder_avg_metric = 0
        self.best_decoder_avg_metric = 0
        num_epochs = self.config["train"]["num_epochs"]
        loss_weights = self.config["train"]["loss_weights"]
        test_freq = self.config["train"]["test_freq"]
        test_mode = self.config["test"]["test_mode"]
        grad_accumulation_steps = self.config["train"]["gradient_accumulation_steps"]

        for epoch in range(restore_epoch+1, num_epochs):
            self.init_test_logit_dict()
            self.model.train()
            epoch_running_loss = 0
            pbar = tqdm(train_train_loader, total=len(train_train_loader))
            
            for batch in pbar:
                loss_dict = self.model.get_training_loss(
                    batch=batch,
                    sem_encoder_ce_criterion=ce_criterion, 
                    sem_encoder_mse_criterion=mse_criterion,
                    bound_encoder_ce_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_bound_criterion=bce_criterion,
                    config=self.config
                )

                total_loss = 0
                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k, v in loss_dict.items():
                        tb_logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item(), step)
                    tb_logger.add_scalar('Train-Total', total_loss.item(), step)

                pbar.set_postfix(loss=f"{total_loss.item():.4f}")
                epoch_running_loss += total_loss.item()

                # backward and gradient accmulation ops
                total_loss /= grad_accumulation_steps
                total_loss.backward()
                if self.config["train"]["gradient_clip"] != 0:  
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["train"]["gradient_clip"])

                if step % grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_loader)
            self.logger.info(f'Epoch {epoch} - Running Loss {epoch_running_loss}')

            schedular.step()

            ################# test #################
            if epoch % test_freq == 0:
                for mode in test_mode:
                    result_dir_test = os.path.join(result_dir, mode)
                    test_result_dict = self.test(
                        test_test_loader, mode, device,
                        result_dir=result_dir_test, model_path=None)
                    
                    avg_metric = sum(test_result_dict.values()) / len(test_result_dict)
                    self.logger.info(f"Testing mode {mode} avg metric value is {avg_metric}")

                    # save best model accoring to avg-metric for different testing modes
                    if mode == "encoder" and avg_metric > self.best_encoder_avg_metric:
                        self.best_encoder_avg_metric = avg_metric
                        state = {
                            'model': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch
                        }
                        torch.save(state, f'{result_dir_test}/best_avg_metric_model.pt')
                        tb_logger.add_scalar(f"{mode}-avg-metric", avg_metric, epoch)

                    elif mode == "decoder" and avg_metric > self.best_decoder_avg_metric:
                        self.best_decoder_avg_metric = avg_metric
                        state = {
                            'model': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch
                        }
                        torch.save(state, f'{result_dir_test}/best_avg_metric_model.pt')
                        tb_logger.add_scalar(f"{mode}-avg-metric", avg_metric, epoch)

                        # save logits for potential postprocessing
                        logit_save_dir = os.path.join(result_dir_test, "average_metric_based", "logits_after_softmax")
                        os.makedirs(logit_save_dir, exist_ok=True)
                        with open(os.path.join(logit_save_dir, "logits.pkl"), 'wb') as f:
                            pickle.dump(self.test_logits_dict, f)

                    # log each test metric into tensorboard writter and train logs
                    for k, v in test_result_dict.items():
                        tb_logger.add_scalar(f'Test-{mode}-{k}', v, epoch)
                        self.logger.info(f"Test-{mode}-{k}-{v}-{epoch}")

                    best_avg_metric = self.best_decoder_avg_metric if mode == "decoder" else self.best_encoder_avg_metric
                    self.logger.info(f"Testing mode {mode} best avg metric value is {best_avg_metric}")
                    self.logger.info("\n\n")
                
                    # save best model accoring to Acc for different testing modes
                    if result_dir_test and 'Acc' in test_result_dict:
                        if "encoder" in mode and test_result_dict["Acc"] > self.best_en_acc:
                            state = {
                                'model': self.model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch
                            }
                            self.best_en_acc = test_result_dict['Acc']
                            torch.save(state, f'{result_dir_test}/best_acc_model.pt')
                        elif "decoder" in mode and test_result_dict["Acc"] > self.best_de_acc:
                            state = {
                                'model': self.model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch
                            }
                            self.best_de_acc = test_result_dict['Acc']
                            torch.save(state, f'{result_dir_test}/best_acc_model.pt')

                            logit_save_dir = os.path.join(result_dir_test, "acc_based", "logits_after_softmax")
                            os.makedirs(logit_save_dir, exist_ok=True)
                            with open(os.path.join(logit_save_dir, "logits.pkl"), 'wb') as f:
                                pickle.dump(self.test_logits_dict, f)
        
        # finish training
        if tb_logger:
            tb_logger.close()

    def test_single_video(self, video_idx, sample, mode, device, model_path, loaded=False):  
        
        assert(mode in ['encoder', 'decoder'])

        if model_path and not loaded:
            self.model.eval()
            self.model.load_state_dict(torch.load(model_path)["model"])
            self.model.to(device)

        # use fixed seed for detersmintic results
        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
        
        with torch.no_grad():
            video_name, video_feats, label = sample["video_name"], sample["video_feats"], sample["f_stage_label"]
            video_feats, label = video_feats.to(device), label.to(device)
            
            # do not support sample_rate, which means that we always feed the whole frame sequence into the model
            if mode == 'encoder':
                out_dict = self.model.forward_condition_model(video_feats) # B T C
                output = out_dict["sem_encoder_out"]
                output = F.softmax(output, 2).cpu()

            if mode == 'decoder':
                output = self.model.ddim_sample(video_feats, seed, reset_rng_after=True)
            
            output = output.squeeze(0).cpu().numpy()
            pred = np.argmax(output, 1)
            label = label.squeeze(0).cpu().numpy()
            
            return video_name, output, pred, label

    def test(self, test_loader, mode, device, result_dir, model_path=None):
        '''
        STEP-1: test single video (make sure test batch_size equals to 1)
        STEP-2: save the preditions to result_dir 
        STEP-3: compute the eval metrics according to all savings in result_dir
        '''

        self.model.eval()
        self.model.to(device)

        loaded = False
        if model_path:
            self.model.load_state_dict(torch.load(model_path)["model"])
            loaded = True # control the test single video not to reload the model again
        
        with torch.no_grad():
            pbar = tqdm(test_loader, total=len(test_loader))
            video_idx = 0 
            for sample in pbar:
                video_name, logits, pred, label = self.test_single_video(
                    video_idx, sample, mode, device, model_path, loaded)
                video_idx += 1
            
                # change numerical predictions into stage names
                pred_events = [self.event_list[int(i)][1] for i in pred]
                gt_events =  [self.event_list[int(i)][1] for i in label]
                
                # make dir to save preditions
                output_dir = os.path.join(result_dir, 'prediction')
                os.makedirs(output_dir, exist_ok=True)
            
                # NOTE: make sure test batch_size equals to 1
                file_name = os.path.join(output_dir, f'{video_name[0]}.csv')
                
                # write predictions and GT labels into csv files
                with open(file_name, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['frame_inds', 'prediction', 'label'])
                    for frame_idx, (logit, pred_label, gt_label) in enumerate(zip(logits, pred_events, gt_events)):
                        writer.writerow([frame_idx, pred_label, gt_label])

                        # collect pred logits for viterbi decoding.
                        self.test_logits_dict[video_name[0]].append((frame_idx, logit, gt_label))

        acc, edit, f1s = func_eval_modified(
            output_dir, test_loader.dataset.video_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }
        
        return result_dict

