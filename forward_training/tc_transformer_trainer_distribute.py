# coding: utf-8

from __future__ import absolute_import, division, print_function

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


class ForwardTextClassificationTrainer:
    def __init__(self, args, device, model, train_dl=None, test_dl=None):
        self.args = args
        self.device = device

        # set data
        self.num_labels = args.num_labels
        self.set_data(train_dl, test_dl)

        # model
        self.model = model
        if self.args.model_type == "distilbert":
            self.model.add_module('pre_classifier',nn.Sequential())
        self.model.to(self.device)

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # freeze
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

        self.grad = None
        self.pert = None

        #??
        #self.update = False

        if self.args.perturbation_sampling and self.args.var_control:
            self.old_grad = None
            self.grad_pool = []

        # var control
        self.grad_for_var_check_list = []
        if self.args.model_type == "distilbert":
            self.layer_id_for_check = 20
        elif self.args.model_type == "bert":
            self.layer_id_for_check = 12
        elif self.args.model_type == "roberta-large":
            self.layer_id_for_check = 12
        elif self.args.model_type == "albert":
            self.layer_id_for_check = 22
        self.var = 0

    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def get_model(self):
        return self.model
    
    # def get_grad(self):
    #     return self.model.grad if self.model.grad is not None else [torch.zeros_like(p) for p in self.model.parameters()]

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl

    def set_perturbation(self, perturbation):
        self.pert = [p.to(self.device, non_blocking=True) for p in perturbation]
        logging.info("Set perturbation in client.")

    # def train_model(self, device=None, args=None):
    #     if not device:
    #         device = self.device

    #     logging.info("train_model self.device: " + str(device))
    #     self.model.to(device)

    #     parameter_number = get_parameter_number(self.model)
    #     logging.info(get_parameter_number(self.model))

    #     trainable_params_number = parameter_number['Trainable']
    #     logging.info("trainable_params_number: " + str(trainable_params_number))
    #     self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)

    #     # training result
    #     global_step = 0
    #     tr_loss, logging_loss = 0.0, 0.0
        
    #     # 批次内扰动数量
    #     v_num = args.v_num if args is not None else self.args.v_num
    #     logging.info("v_num: " + str(v_num))
    #     logging.info("data_id: " + str(args.data_id))
    #     if args.data_id == 0:
    #         self.grad_zoo = [torch.zeros_like(p) for p in self.params]

    #     coeff_pert_p = None
    #     coeff_pert_v = None
    #     coeff_v_only = None
    #     logging.info(f"Self.pert == None?: {self.pert == None} !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    #     if (self.pert is not None) and (args.round_idx != 0):
    #         coeff_pert_p = args.beta * math.sqrt((1 - args.alpha) / args.pool_size)
    #         coeff_pert_v = args.beta * math.sqrt(args.alpha/trainable_params_number)
    #         logging.info("Successfully load hybrid perturbation from cloud.\n")
    #     else:
    #         coeff_v_only = args.beta * math.sqrt(1.0 / trainable_params_number)
    #         logging.info("Successfully load local perturbation from cloud.\n")
        

    #     if self.args.perturbation_sampling:
    #         v_num = len(self.train_dl)
    #         v_buffer = {}
    #         index = 0
    #         if self.args.var_control:
    #             self.grad = self.old_grad
    #         for k,v in self.model.named_parameters():
    #             # logging.info(index)
    #             if self.grad != None and v.requires_grad:
    #                 # logging.info("generate v")
    #                 shape = v.shape
    #                 candidate_v = torch.randn((v_num*10,*shape),device="cpu")
    #                 target_grad = self.grad[index]

    #                 # logging.info("flatten")
    #                 target_grad = torch.flatten(target_grad)
    #                 candidate_v = torch.flatten(candidate_v,start_dim=1)

    #                 cos_sim = calculate_cos_sim(candidate_v,target_grad,device)
    #                 sorted_values, sorted_indices = torch.sort(cos_sim, descending=True)

    #                 v_buffer[index] = [candidate_v[i].reshape(v.shape) for i in sorted_indices[:v_num]]
    #             index += 1

    #     self.grad = [torch.zeros_like(p) for p in self.params]

    #     with torch.no_grad():
    #         logging.info("Start training with Forward Gradient. epoch=" + str(self.args.epochs))
    #         for epoch in range(0, self.args.epochs):
    #             for batch_idx, batch in enumerate(self.train_dl):

    #                 batch = tuple(t for t in batch)
    #                 x = batch[1].to(device)
    #                 labels = batch[4].to(device)

    #                 # 优化函数
    #                 f = partial(
    #                     functional_get_loss,
    #                     model=self.fmodel,
    #                     buffers = self.buffers,
    #                     num_classes = self.num_labels,
    #                     x=x,
    #                     t=labels,
    #                 )
                    
    #                 # 生成扰动
    #                 if self.args.perturbation_sampling and v_buffer != {}:
    #                     v_params = tuple([v_buffer[i][batch_idx].to(device) if p.requires_grad == True else torch.zeros_like(p) for i,p in enumerate(self.params)])
    #                 else:
    #                     v_params = tuple([torch.randn_like(p) if p.requires_grad == True else torch.zeros_like(p) for p in self.params])

                   
    #                 if (self.pert is not None) and (args.round_idx != 0):
    #                     final_perturbation = [
    #                         coeff_pert_p * p + coeff_pert_v * v
    #                         for p, v in zip(self.pert, v_params)
    #                     ]
    #                 else:
    #                     final_perturbation = [coeff_v_only * v for v in v_params]
                    
    #                 # 计算方向导数
    #                 loss, jvp = calculate_jvp(f, self.params, final_perturbation)
                    
    #                 # 计算梯度
    #                 for j, fg in enumerate(self.grad):
    #                     fg.add_(jvp*final_perturbation[j])
    #                     if self.args.var_control and j == self.layer_id_for_check:
    #                         self.grad_for_var_check_list.append(jvp*final_perturbation[j])


    #                 current_loss = loss.item()
    #                 logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
    #                                                                         len(self.train_dl), current_loss))

    #                 global_step += 1
    #                 if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
    #                                                             and global_step!=0  and global_step % self.args.evaluate_during_training_steps == 0):
    #                     results, _, _ = self.eval_model(epoch, global_step)

    #                 if self.args.is_debug_mode == 1 and global_step > 3:
    #                     break

    #     if self.args.var_control:
    #         self.var = calculate_var(self.grad_for_var_check_list,)
    #         logging.info(f"num of fwdgrad: {len(self.grad_for_var_check_list)}, var: {self.var}")
    #         if self.args.perturbation_sampling:
    #             self.grad_pool.append(self.grad)
    #     return global_step, tr_loss / global_step

    def train_model(self, device=None, args=None):
        if not device:
            device = self.device

        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        logging.info(get_parameter_number(self.model))
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.perturbation_sampling:
            v_num = len(self.train_dl)
            v_buffer = {}
            index = 0
            if self.args.var_control:
                self.grad = self.old_grad
            for k,v in self.model.named_parameters():
                # logging.info(index)
                if self.grad != None and v.requires_grad:
                    # logging.info("generate v")
                    shape = v.shape
                    candidate_v = torch.randn((v_num*10,*shape),device="cpu")
                    target_grad = self.grad[index]

                    # logging.info("flatten")
                    target_grad = torch.flatten(target_grad)
                    candidate_v = torch.flatten(candidate_v,start_dim=1)

                    cos_sim = calculate_cos_sim(candidate_v,target_grad,device)
                    sorted_values, sorted_indices = torch.sort(cos_sim, descending=True)

                    v_buffer[index] = [candidate_v[i].reshape(v.shape) for i in sorted_indices[:v_num]]
                index += 1

        self.grad = [torch.zeros_like(p) for p in self.params]
        logging.info("self.grad is not None: " + str(self.grad is not None))

        coeff_pert_p = None
        coeff_pert_v = None
        coeff_v_only = None
        if (self.pert is not None) and (args.round_idx != 0):
            coeff_pert_p = args.beta * math.sqrt((1 - args.alpha) / args.pool_size)
            coeff_pert_v = args.beta * math.sqrt(args.alpha)#trainable_params_number
        else:
            coeff_v_only = args.beta * math.sqrt(1.0)#trainable_params_number

        with torch.no_grad():
            logging.info("Start training with Forward Gradient. epoch=" + str(self.args.epochs))

            for epoch in range(0, self.args.epochs):
                for batch_idx, batch in enumerate(self.train_dl):

                    batch = tuple(t for t in batch)
                    x = batch[1].to(device)
                    labels = batch[4].to(device)

                    # 优化函数
                    f = partial(
                        functional_get_loss,
                        model=self.fmodel,
                        buffers = self.buffers,
                        num_classes = self.num_labels,
                        x=x,
                        t=labels,
                    )

                    # 生成扰动
                    if self.args.perturbation_sampling and v_buffer != {}:
                        v_params = tuple([v_buffer[i][batch_idx].to(device) if p.requires_grad == True else torch.zeros_like(p) for i,p in enumerate(self.params)])
                    else:
                        v_params = tuple([torch.randn_like(p) if p.requires_grad == True else torch.zeros_like(p) for p in self.params])
                    # 定义函数计算非零元素的绝对均值
                    def mean_abs_nonzero(tensor):
                        flat_nonzero_abs = torch.cat([p.flatten() for p in tensor]).abs()
                        flat_nonzero_abs = flat_nonzero_abs[flat_nonzero_abs != 0]
                        return flat_nonzero_abs.mean().item() if flat_nonzero_abs.numel() > 0 else 0.0
                    
                    if (self.pert is not None) and (args.round_idx != 0):
                        # logging.info("scale of bp_perturbation: " + str(mean_abs_nonzero(self.pert)))
                        # logging.info("scale of v_params: " + str(mean_abs_nonzero(v_params)))
                        final_perturbation = [
                            coeff_pert_p * p + coeff_pert_v * v
                            for p, v in zip(self.pert, v_params)
                        ]
                        # logging.info("scale of final_perturbation: " + str(mean_abs_nonzero(final_perturbation)))
                    else:
                        final_perturbation = [coeff_v_only * v for v in v_params]
                        # logging.info("scale of final_perturbation 1: " + str(mean_abs_nonzero(final_perturbation)))

                    
                    # === scale: 只考虑非零位置 ===
                    # def normalize_to_target(vec_list, target):
                    #     flat = torch.cat([v.flatten() for v in vec_list])
                    #     flat = flat[flat != 0]
                    #     if flat.numel() == 0:
                    #         return vec_list  # 全 0 就直接返回
                    #     scale = target / (flat.abs().mean() + 1e-8)
                    #     return [v * scale for v in vec_list]

                    # # 归一化 self.pert 和 v_params
                    # if (self.pert is not None) and (args.round_idx != 0):
                    #     pert_scaled = normalize_to_target(self.pert, target=0.1)
                    #     v_scaled    = normalize_to_target(v_params, target=1.0)

                    #     final_perturbation = [
                    #         coeff_pert_p * p + coeff_pert_v * v
                    #         for p, v in zip(pert_scaled, v_scaled)
                    #     ]
                    # else:
                    #     v_scaled = normalize_to_target(v_params, target=1.0)
                    #     final_perturbation = [coeff_v_only * v for v in v_scaled]

                        
                    # 计算方向导数
                    loss, jvp = calculate_jvp(f, self.params, final_perturbation)
                    
                    # 计算梯度
                    for j, fg in enumerate(self.grad):
                        fg.add_(jvp * final_perturbation[j])
                        if self.args.var_control and j == self.layer_id_for_check:
                            self.grad_for_var_check_list.append(jvp * final_perturbation[j])


                    current_loss = loss.item()
                    logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                            len(self.train_dl), current_loss))

                    global_step += 1
                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                                and global_step!=0  and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(epoch, global_step)

                    if self.args.is_debug_mode == 1 and global_step > 3:
                        break

        if self.args.var_control:
            self.var = calculate_var(self.grad_for_var_check_list,)
            logging.info(f"num of fwdgrad: {len(self.grad_for_var_check_list)}, var: {self.var}")
            if self.args.perturbation_sampling:
                self.grad_pool.append(self.grad)
        return global_step, tr_loss / global_step


    # def train_model_bp(self, device=None):
    #     if not device:
    #         device = self.device

    #     logging.info("train_model self.device: " + str(device))
    #     self.model.to(device)

    #     logging.info(get_parameter_number(self.model))
    #     self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)

    #     # training result
    #     global_step = 0
    #     tr_loss, logging_loss = 0.0, 0.0

    #     if self.args.fl_algorithm == "FedProx":
    #         global_model = copy.deepcopy(self.model)

    #     self.grad = [torch.zeros_like(p) for p in self.params]

    #     for epoch in range(0, self.args.epochs):

    #         for batch_idx, batch in enumerate(self.train_dl):
    #             self.model.train()
    #             batch = tuple(t for t in batch)
    #             # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #             x = batch[1].to(device)
    #             labels = batch[4].to(device)

    #             output = self.model(x)
                
    #             logits = output[0]

    #             loss_fct = CrossEntropyLoss()
    #             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                

    #             if self.args.fl_algorithm == "FedProx":
    #                 fed_prox_reg = 0.0
    #                 mu = self.args.fedprox_mu
    #                 for (p, g_p) in zip(self.model.parameters(),
    #                                     global_model.parameters()):
    #                     fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
    #                 loss += fed_prox_reg

    #             current_loss = loss.item()
    #             logging.info("Training with BP in the cloud: epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
    #                                                                        len(self.train_dl), current_loss))

    #             if self.args.gradient_accumulation_steps > 1:
    #                 loss = loss / self.args.gradient_accumulation_steps

    #             loss.backward()
    #             for i,p in enumerate(self.model.parameters()):
    #                 if p.grad is not None:
    #                     self.grad[i] += copy.deepcopy(p.grad.data)
    #             self.model.zero_grad()

    #             if self.args.is_debug_mode == 1 and global_step > 3:
    #                 break
        
    #     return global_step, tr_loss
        
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

        self.grad_bp = [torch.zeros_like(p) for p in self.params]

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
                logging.info("Training with BP in the cloud: epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), current_loss))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                for i,p in enumerate(self.model.parameters()):
                    if p.grad is not None:
                        self.grad_bp[i] += copy.deepcopy(p.grad.data)
                self.model.zero_grad()

                if self.args.is_debug_mode == 1 and global_step > 3:
                    break
        
            num_batches = len(self.train_dl)
            for i in range(len(self.grad_bp)):
                self.grad_bp[i] /= num_batches

        return global_step, tr_loss
        
    # def eval_model(self, epoch=0, global_step=0, device=None):
    #     if not device:
    #         device = self.device

    #     results = {}

    #     eval_loss = 0.0
    #     nb_eval_steps = 0
    #     n_batches = len(self.test_dl)
    #     test_sample_len = len(self.test_dl.dataset)
    #     preds = np.empty((test_sample_len, self.num_labels))

    #     out_label_ids = np.empty(test_sample_len)
    #     self.model.to(device)
    #     self.model.eval()
    #     self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
    #     logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
    #     for i, batch in enumerate(self.test_dl):
    #         with torch.no_grad():
    #             batch = tuple(t.to(device) for t in batch)
    #             x = batch[1]
    #             labels = batch[4]

    #             output = self.model(x)
    #             logits = output[0]

    #             loss_fct = CrossEntropyLoss()
    #             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    #             eval_loss += loss.item()

    #         nb_eval_steps += 1
    #         start_index = self.args.eval_batch_size * i

    #         end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
    #         preds[start_index:end_index] = logits.detach().cpu().numpy()
    #         out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

    #     eval_loss = eval_loss / nb_eval_steps

    #     model_outputs = preds
    #     preds = np.argmax(preds, axis=1)
    #     result, wrong = self.compute_metrics(preds, out_label_ids, self.test_dl.examples)
    #     result["eval_loss"] = eval_loss
    #     results.update(result)

    #     self.results.update(result)
    #     logging.info(self.results)

    #     return result, model_outputs, wrong

    def eval_model(self, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = len(self.test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        for i, batch in enumerate(self.test_dl):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)
                x = batch[1]
                labels = batch[4]

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(preds, out_label_ids, self.test_dl.examples)
        result["eval_loss"] = eval_loss
        results.update(result)

        self.results.update(result)
        logging.info(self.results)

        return result, model_outputs, wrong
    
    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

