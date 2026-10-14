import torch
import torch.nn as nn
import logging
from collections import defaultdict
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import func_eval_stage1


class Trainer:
    def __init__(self, config, model, logger=None, writter=None, device=None):
        self.config = config
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        event_list =  np.loadtxt(config["development_stage_list"], dtype=str)
        self.event_list = [e[1] for e in event_list]
        self.model = model
        
        self.best_acc = 0.0 
        self.best_test_acc = 0.0
        self.best_avg_metric = 0.0

        self.logger = logger
        self.writter = writter
        
    def collect_preds(self, mode, video_name, frame_ind, pred, label):
        if mode == "train":
            for idx in range(len(video_name)):
                self.train_res_dict[video_name[idx]].append((frame_ind[idx], self.event_list[pred[idx]], self.event_list[label[idx]]))
        else:
            for idx in range(len(video_name)):
                self.test_res_dict[video_name[idx]].append((frame_ind[idx], self.event_list[pred[idx]], self.event_list[label[idx]]))            

    def collect_logits(self, mode, video_name, frame_ind, logits, label):
        if mode == "train":
            for idx in range(len(video_name)):
                self.train_logit_dict[video_name[idx]].append((frame_ind[idx], logits[idx], self.event_list[label[idx]]))
        else:
            for idx in range(len(video_name)):
                self.test_logit_dict[video_name[idx]].append((frame_ind[idx], logits[idx], self.event_list[label[idx]]))

    def init_pred_dict(self, mode="train"):
        if mode == "train":
            self.train_res_dict = defaultdict(list)
            self.train_logit_dict = defaultdict(list)
        elif mode == "test":
            self.test_res_dict = defaultdict(list)
            self.test_logit_dict = defaultdict(list)

    def save_best_model(self, epoch, acc):
        if acc > self.best_test_acc:
            self.best_test_acc = acc
            model_path = os.path.join(
                self.config["model_save_dir"],
                f"best_model.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            logging.info(f"Saved best model to {model_path}, epoch-{epoch}.")


    def save_best_predictions(self, acc=None):
        if acc > self.best_test_acc:
            for vid in self.test_res_dict:
                results = self.test_res_dict[vid]
                if not results:
                    continue
                frame_inds = [r[0] for r in results]
                predictions = [r[1] for r in results]
                labels = [r[2] for r in results]
                df = pd.DataFrame({
                    'frame_ind': frame_inds,
                    'prediction': predictions,
                    'label': labels
                })
                csv_path = os.path.join(self.config["pred_save_dir"], f"{vid}.csv")
                df.to_csv(csv_path, index=False)
                logging.info(f"Updated best predictions to {csv_path}")


    def train(self, train_loader, optimizer, schedular=None, test_loader=None, video_list=None):
        mode = "train"
        criterion = nn.CrossEntropyLoss(weight=None)

        num_epochs = self.config["num_epochs"]
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            self.init_pred_dict(mode=mode)
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
            for batch in batch_iter:
                video_name = batch["video_name"]
                frame = batch["frame"]
                frame_ind = batch["frame_ind"]
                stage_label = batch["f_stage_label"]

                frame = frame.to(self.device, non_blocking=True)
                stage_label = stage_label.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = self.model(frame)
                loss = criterion(outputs, stage_label)
                loss.backward()
                optimizer.step()

                batch_iter.set_postfix(loss=f"{loss.cpu().item():.4f}")
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += stage_label.size(0)
                correct += predicted.eq(stage_label).sum().item()

                self.collect_preds(mode, video_name, frame_ind.cpu().numpy(), predicted.cpu().numpy(), stage_label.cpu().numpy())
            
            schedular.step()

            epoch_loss = running_loss / len(train_loader)
            self.logger.info(f"Epoch running loss - Epoch: {epoch} /  Loss: {epoch_loss}")
            self.writter.add_scalar('Loss/train', epoch_loss, epoch)

            log_preq = self.config["log_freq"]
            if epoch == 0 or epoch % log_preq == 0 or epoch == self.config["num_epochs"]:

                for vid in self.train_res_dict:
                    sorted_results = sorted(self.train_res_dict[vid], key=lambda x: x[0])
                    self.train_res_dict[vid] = sorted_results

                acc, edit, f1s = func_eval_stage1(self.train_res_dict)
                result_dict = {
                    'Acc': acc,
                    'Edit': edit,
                    'F1@10': f1s[0],
                    'F1@25': f1s[1],
                    'F1@50': f1s[2]
                    }
                
                self.logger.info(f"Evaluation Metrics: Acc={result_dict['Acc']:.2f}, Edit={result_dict['Edit']:.2f}")
                self.logger.info(f"F1@10={result_dict['F1@10']:.2f}, F1@25={result_dict['F1@25']:.2f}, F1@50={result_dict['F1@50']:.2f}")
            
                # log to TensorBoard
                self.writter.add_scalar('Acc/train', result_dict['Acc'], epoch)
                self.writter.add_scalar('Edit/train', result_dict['Edit'], epoch)
                self.writter.add_scalar('F1@10/train', result_dict['F1@10'], epoch)
                self.writter.add_scalar('F1@25/train', result_dict['F1@25'], epoch)
                self.writter.add_scalar('F1@50/train', result_dict['F1@50'], epoch)

            test_freq = self.config["test_freq"]
            if epoch == 0 or epoch % test_freq == 0 or epoch == self.config["num_epochs"]:
                if test_loader:
                    self.test(epoch, test_loader)


    def test(self, epoch, test_loader):
        mode = "test"
        self.init_pred_dict(mode=mode)
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        
        with torch.no_grad():
            batch_iter = tqdm(test_loader, desc=f"Testing", unit="batch")          
            for batch in batch_iter:
                video_name = batch["video_name"]
                frame_ind = batch["frame_ind"]
                frame = batch["frame"].to(self.device, non_blocking=True)
                stage_label = batch["f_stage_label"].to(self.device, non_blocking=True)
                
                outputs = self.model(frame)
                loss = criterion(outputs, stage_label)
                running_loss += loss.item()

                _, predicted = outputs.max(1)

                if self.config["use_viterbi_decode"]:
                    self.collect_logits(
                        mode=mode,
                        video_name=video_name,
                        frame_ind=frame_ind.cpu().numpy(),
                        logits=outputs.cpu().numpy(),
                        label=stage_label.cpu().numpy()
                    )

                self.collect_preds(
                    mode=mode,
                    video_name=video_name,
                    frame_ind=frame_ind.cpu().numpy(),
                    pred=predicted.cpu().numpy(),
                    label=stage_label.cpu().numpy()
                )
        
        epoch_loss = running_loss / len(test_loader)
        for vid in self.test_res_dict:
            sorted_results = sorted(self.test_res_dict[vid], key=lambda x: x[0])
            self.test_res_dict[vid] = sorted_results
        acc, edit, f1s = func_eval_stage1(self.test_res_dict)

        
        if self.config["use_viterbi_decode"] and acc > self.best_acc:
            for vid in self.test_logit_dict:
                sorted_item = sorted(self.test_logit_dict[vid], key=lambda x: x[0])
                self.test_logit_dict[vid] = sorted_item
            with open(os.path.join(self.config["pred_save_dir"], "logits.pkl"), 'wb') as f:
                pickle.dump(self.test_logit_dict, f)

        self.save_best_predictions(acc)
        self.save_best_model(epoch=epoch, acc=acc)    
        
        self.logger.info("\n\n")
        self.logger.info("===== Test Evaluation =====")
        self.logger.info(f"Test Loss: {epoch_loss:.4f}")
        self.logger.info(f"Test Metrics: Acc={acc:.2f}, Edit={edit:.2f}")
        self.logger.info(f"F1@10={f1s[0]:.2f}, F1@25={f1s[1]:.2f}, F1@50={f1s[2]:.2f}")

        self.writter.add_scalar('Loss/test', epoch_loss, epoch) 
        self.writter.add_scalar('Acc/test', acc, epoch)
        self.writter.add_scalar('Edit/test', edit, epoch)
        self.writter.add_scalar('F1@10/test', f1s[0], epoch)
        self.writter.add_scalar('F1@25/test', f1s[1], epoch)
        self.writter.add_scalar('F1@50/test', f1s[2], epoch)

        return {
            'loss': epoch_loss,
            'test_acc': acc,
            'edit': edit,
            'f1s': f1s
        }



