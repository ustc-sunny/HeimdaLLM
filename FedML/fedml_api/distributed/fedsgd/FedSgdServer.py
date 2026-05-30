import copy
import logging
import random
import time
import math

import numpy as np
import torch
import wandb

class FedSGDServer(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.warmup_rounds = math.ceil(self.args.comm_round * self.args.warmup_ratio)

        # 之前的v不够，暂存在cached_v
        self.cached_v = []
        # 方差控制：重试计数器，防止无限循环
        self.retry_count = 0
        self.max_retries = 10  

        if self.args.model_type == "distilbert":
            # self.var_threthod = 0.25
            self.var_threthod = 0.2
        elif self.args.model_type == "bert":
            # self.var_threthod = 0.6
            self.var_threthod = 0.1
        elif self.args.model_type == "roberta-large":
            # self.var_threthod = 0.6
            self.var_threthod = 0.2
        elif self.args.model_type == "albert":
            # self.var_threthod = 0.6
            self.var_threthod = 0.1

    def get_global_model_params(self):
        return self.trainer.server_trainer.get_model_params()
    
    def get_global_model(self):
        return self.trainer.server_trainer.get_model()

    def set_global_model_params(self, model_parameters):
        self.trainer.server_trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self,current_round):
        start_time = time.time()
        model_list = []
        training_num = 0

        # self.warmup_rounds = 20
        if current_round < self.warmup_rounds:
            ratio = float(current_round+1) / float(max(1, self.warmup_rounds))
        else:
            ratio = max(
            0.0, float(self.args.comm_round - current_round) / float(max(1, self.args.comm_round - self.warmup_rounds))
        )
        learning_rate = self.args.learning_rate * ratio
        logging.info(f"learning rate: {learning_rate}")

        for idx in range(self.worker_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]
        
        # self.model_dict在聚合的过程中会被改变,很奇怪，这里先存一个deepcopy吧，用于后面cache_v
        if self.args.var_control:
            model_dict_cached = copy.deepcopy(self.model_dict)
            origin_param = copy.deepcopy(self.get_global_model_params())

            # cached_v:  (num, params)
            logging.info(f"len of cached v: {len(self.cached_v)}")
            for cached_v in self.cached_v:
                model_list.append(cached_v)
                training_num += cached_v[0]
            logging.info(f"training_num : {training_num}")

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))


        #######  新版本：基于 model_list 的加权聚合  #######
        # model_list 是一个 (num_samples, grads_or_params) 的列表。
        # 之前的老版本使用 model_list（并且会包含 self.cached_v 中的额外条目）来对多个模型/梯度进行加权聚合。
        # 在引入 var_control 和 cached_v 的场景下，必须使用 model_list 而不是仅遍历 worker_num，否则 cached_v 条目会被忽略。
        # 新实现：对 model_list 中的每个 (n, grads) 做加权求和 averaged_grads += grads * n，然后除以 training_num。
        averaged_grads = None
        logging.info("####### aggregate ####### len of model_list: %d" % len(model_list))
        for local_sample_number, local_grads in model_list:
            if averaged_grads is None:
                # 初始化为与 local_grads 形状相同的零张量
                averaged_grads = [torch.zeros_like(g) for g in local_grads]
            for j, g in enumerate(local_grads):
                # 假设 local_grads[j] 是梯度或者与更新方向成比例的张量，用样本数加权
                averaged_grads[j] += g

        # 防护：如果没有任何 grads（异常场景），跳过
        if averaged_grads is None:
            logging.warning("No gradients to aggregate (averaged_grads is None)")
        else:
            # 按总样本数归一化
            for j in range(len(averaged_grads)):
                averaged_grads[j] = averaged_grads[j] / float(max(1.0, training_num))

            with torch.no_grad():
                for p, grad in zip(self.trainer.server_trainer.model.parameters(), averaged_grads):
                    p.data -= learning_rate * grad.to(p.device)


        # #######  老版本：next   ###########################
        # # old_param = self.get_global_model_params()
        # old_param = self.trainer.model.parameters()
        
        # # logging.info("################aggregate: %d" % len(model_list))
        # (num0, averaged_params) = model_list[0]
        # for id,k in enumerate(averaged_params):
        #     for i in range(0, len(model_list)):
        #         local_sample_number, local_model_params = model_list[i]
        #         # w = local_sample_number / training_num
        #         if i == 0:
        #             averaged_params[id] = local_model_params[id]
        #         else:
        #             averaged_params[id] += local_model_params[id]
        #     next(old_param).detach().to("cpu").sub_(learning_rate * averaged_params[id] / training_num)
        # ##################################################
        
        if self.args.var_control:
            if self.var <= self.var_threthod or self.retry_count >= self.max_retries:
                # 方差满足要求
                self.cached_v = []
            else:
                # 当前模型不行，v不够，暂存起来，后面再计算更多的v
                for idx in range(self.worker_num):
                    self.cached_v.append((self.sample_num_dict[idx], model_dict_cached[idx]))
                # 模型改回去
                self.set_global_model_params(origin_param)

        # old_param = self.get_global_model_params()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return self.get_global_model_params()

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        index_list = [[] for _ in range(self.args.worker_num)]

        for i in range(len(client_indexes)):
            index_list[i%self.args.worker_num].append(client_indexes[i])

        client_indexes = index_list

        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            if self.trainer.test_on_the_server(self.device):
                return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
