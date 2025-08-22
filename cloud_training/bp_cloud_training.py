# cloud/cloud_bp_training.py
import copy
from hashlib import shake_128
import logging

import numpy as np
import sklearn
import torch
from torch import nn
from training.utils.text_classification_utils import *
from forward_training.utils.fwdgrad_utils import *
from torch.nn import CrossEntropyLoss
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from functools import partial
import functorch as fc
from torch.cuda.amp import autocast

class CloudBackpropController:
    """
    仅运行在 Cloud 侧：
    1. 接收 server 发来的聚合参数；
    2. 在 Cloud 私有数据上做若干 epoch 的反向训练；
    3. 返回“修正后的梯度”或“修正后的参数增量”给 server。
    """
    def __init__(self, model, args, train_dl, test_dl=None, device="cuda"):
        self.model = copy.deepcopy(model)      # Cloud 自己维护一份参数
        self.args  = args
        self.train_dl = train_dl               # Cloud 私有数据
        self.test_dl  = test_dl
        self.device   = device
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.cloud_learning_rate,
            weight_decay=args.weight_decay
        )
        total_steps = len(train_dl) * args.cloud_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )


    ## 增加 BP 训练
    def train_model_bp(self, device=None):
        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        logging.info(get_parameter_number(self.model))
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)

        self.grad = [torch.zeros_like(p) for p in self.params]

        for epoch in range(0, self.args.epochs):

            for batch_idx, batch in enumerate(self.train_dl):
                self.model.train()
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                output = self.model(x)
                
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(self.model.parameters(),
                                        global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                current_loss = loss.item()
                logging.info("Training with BP in the server: epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), current_loss))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                for i,p in enumerate(self.model.parameters()):
                    if p.grad is not None:
                        self.grad[i] += copy.deepcopy(p.grad.data)
                self.model.zero_grad()

                if self.args.is_debug_mode == 1 and global_step > 3:
                    break
        
        return global_step, tr_loss
        
    # ---------- 供 Server 调用 ----------
    def cloud_train_step(self, state_dict_from_server: dict):
        """
        state_dict_from_server: server 聚合后的全局权重
        返回：delta_dict = cloud_train后的权重 - 接收的权重
        """
        self.model.load_state_dict(state_dict_from_server)

        self.model.train()
        for epoch in range(self.args.cloud_epochs):
            for batch in self.train_dl:
                batch = tuple(t.to(self.device) for t in batch)
                x, labels = batch[1], batch[4]

                self.optimizer.zero_grad()
                logits = self.model(x)[0]
                loss = CrossEntropyLoss()(logits.view(-1, self.args.num_labels),
                                          labels.view(-1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        # 计算 delta
        delta_dict = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                delta_dict[name] = p.cpu() - state_dict_from_server[name].cpu()
        return delta_dict