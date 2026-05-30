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

        self.pool_size = args.pool_size
        self.model_pool = []

    def update_model(self, weights):
        self.trainer.cloud_trainer.set_model_params(weights)

    # def update_dataset(self, client_index):
        # self.client_index = client_index
        # self.train_local = [self.train_data_cloud_dict[id] for id in client_index]
        # self.local_sample_number = self.train_data_cloud_num_dict[client_index[0]]
        # self.test_local = self.test_data_cloud_dict[client_index[0]]

        # self.train_local_list = [[data for data in self.train_local[i]] for i in range(len(self.train_local))]

    
    
    def train_model_bp(self):
        self.trainer.train_bp(self.train_global, self.device, self.args)
        logging.info("Cloud: finish backpropagation training")

        # grads = self.trainer.get_grad()
        weights = [para.detach().cpu() for para in self.trainer.cloud_trainer.grad_bp]
        logging.info("Cloud: get model gradients")
        
        if len(self.model_pool) >= self.pool_size:
            self.model_pool.pop(0)
        self.model_pool.append(weights)
        logging.info("Cloud: append model gradients to pool, pool size = " + str(len(self.model_pool)))
        return weights

    # def create_perturbation(self):
    #     # print("DEBUG: create_perturbation called", flush=True)
    #     alpha = torch.randn(len(self.model_pool), device=self.device)
    #     alpha = alpha / (alpha.norm() + 1e-8)
    #     # print("DEBUG: alpha=" + str(alpha), flush=True)
    #     logging.info("Cloud: sample alpha." + str(alpha))
    #     perturbation = [torch.zeros_like(p, device=self.device) for p in self.model_pool[0]]
        
    #     for i, grad_list in enumerate(self.model_pool):
    #         # print("DEBUG: grad_list=" + str(grad_list), flush=True)
    #         for j, g in enumerate(grad_list):
    #             perturbation[j] += alpha[i].item() * g.to(self.device)
    #             # print("DEBUG: perturbation=" + str(perturbation[j]), flush=True)
    #     logging.info("Cloud: create perturbation.")
    #     return perturbation
    
    def create_perturbation(self):
        scale = 10000.0
        # print("DEBUG: create_perturbation called", flush=True)
        alpha = torch.randn(len(self.model_pool), device=self.device)
        alpha = alpha / (alpha.norm() + 1e-8)
        # print("DEBUG: alpha=" + str(alpha), flush=True)
        logging.info("Cloud: sample alpha." + str(alpha))
        perturbation = [torch.zeros_like(p, device=self.device) for p in self.model_pool[0]]
        
        for i, grad_list in enumerate(self.model_pool):
            # print("DEBUG: grad_list=" + str(grad_list), flush=True)
            for j, g in enumerate(grad_list):
                perturbation[j] += alpha[i].item() * g.to(self.device)
                # print("DEBUG: perturbation=" + str(perturbation[j]), flush=True)
        logging.info("Cloud: create perturbation.")
        
        # === 采样 100 个非零值 ===
        flat_v = torch.cat([p.flatten() for p in perturbation])
        flat_nonzero = flat_v[flat_v != 0].abs()          # 过滤 0
        sample_cnt = min(100, flat_nonzero.numel())
        sample_vals = flat_nonzero[torch.randperm(flat_nonzero.numel())[:sample_cnt]]
        mean_val = sample_vals.mean().item() if sample_cnt > 0 else 0.0

        logging.info(f"[Cloud] perturbation NONZERO sample(100): {sample_vals.cpu().tolist()}")
        logging.info(f"[Cloud] perturbation NONZERO mean(100):  {mean_val:.6f}")

        perturbation = [p * scale for p in perturbation]
        
        # 1. 非零绝对值
        flat_nonzero_abs = torch.cat([p.flatten() for p in perturbation]).abs()
        flat_nonzero_abs = flat_nonzero_abs[flat_nonzero_abs != 0]

        cloud_nonzero_mean = flat_nonzero_abs.mean().item() if flat_nonzero_abs.numel() > 0 else 0.0
        logging.info(f"[Cloud] after align, non-zero abs mean={cloud_nonzero_mean:.4f}")

        # if torch.isnan(torch.tensor(cloud_nonzero_mean)):
        #     perturbation = [0.4 for p in perturbation]
        # else:
        scale = 0.01/cloud_nonzero_mean
        perturbation = [p * scale for p in perturbation]

        # # 1. 非零绝对值
        # flat_nonzero_abs = torch.cat([p.flatten() for p in perturbation]).abs()
        # flat_nonzero_abs = flat_nonzero_abs[flat_nonzero_abs != 0]

        # cloud_nonzero_mean = flat_nonzero_abs.mean().item() if flat_nonzero_abs.numel() > 0 else 0.0
        # logging.info(f"[Cloud1] after align, non-zero abs mean={cloud_nonzero_mean:.4f}")

        return perturbation
    