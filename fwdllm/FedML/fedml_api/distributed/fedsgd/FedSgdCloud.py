import logging
import torch


class FedSGDCloud(object):

    def __init__(self, train_data_cloud,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer
        self.train_global = train_data_cloud

        # self.train_data_cloud_dict = train_data_cloud_dict
        # self.train_data_cloud_num_dict = train_data_cloud_num_dict
        # self.test_data_cloud_dict = test_data_cloud_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args
        self.accumulated_error = None

        self.pool_size = 2
        self.model_pool = []

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    # def update_dataset(self, client_index):
        # self.client_index = client_index
        # self.train_local = [self.train_data_cloud_dict[id] for id in client_index]
        # self.local_sample_number = self.train_data_cloud_num_dict[client_index[0]]
        # self.test_local = self.test_data_cloud_dict[client_index[0]]

        # self.train_local_list = [[data for data in self.train_local[i]] for i in range(len(self.train_local))]

    
    
    def train_model_bp(self):
        self.trainer.train_bp(self.train_global, self.device, self.args)

        # grads = self.trainer.get_grad()
        weights = [para.detach().cpu() for para in self.trainer.model_trainer.grad]
        logging.info("Cloud: get model gradients")
        
        if len(self.model_pool) >= self.pool_size:
            self.model_pool.pop(0)
        self.model_pool.append(weights)
        self.round_idx += 1
        return weights

    def create_perturbation(self):
        alpha = torch.randn(len(self.model_pool), device=self.device)
        perturbation = [torch.zeros_like(p) for p in self.model_pool[0]]
        for i, grad_list in enumerate(self.model_pool):
            for j, g in enumerate(grad_list):
                perturbation[j] += alpha[i] * g.to(self.device)
        logging.info("Create perturbation in cloud.")
        return perturbation
    