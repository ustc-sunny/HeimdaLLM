import logging
import math
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
        self.direction_counter = 0

    def update_model(self, weights):
        self.trainer.cloud_trainer.set_model_params(weights)

    @staticmethod
    def create_sync_placeholder():
        """Return a non-None, zero-payload round barrier for alpha=1."""
        logging.info(
            "[ZGR] alpha=1: send an empty synchronization sentinel (no cloud tensors)"
        )
        return []

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
        # The wire format contains one direction rather than an orthonormal
        # basis. Supporting m > 1 without transmitting that basis would not
        # implement Eq. (8), so reject it instead of silently mis-scaling it.
        if self.pool_size != 1:
            raise NotImplementedError(
                "the current GGD wire format supports only pool_size=1; "
                "an orthonormal basis is required for pool_size>1"
            )
        if not self.model_pool:
            raise ValueError("cannot create cloud perturbation from an empty gradient pool")
        if len(self.model_pool) != 1:
            raise ValueError(
                "pool_size=1 requires exactly one cloud gradient, got %d"
                % len(self.model_pool)
            )

        gradient = self.model_pool[0]
        if not gradient:
            raise ValueError("cannot create cloud perturbation from an empty gradient list")

        squared_norms = []
        for tensor_index, tensor in enumerate(gradient):
            if tensor.numel() == 0:
                continue
            if not torch.isfinite(tensor).all().item():
                raise FloatingPointError(
                    "cloud gradient tensor %d contains non-finite values"
                    % tensor_index
                )
            tensor_norm = tensor.detach().float().norm().item()
            if not math.isfinite(tensor_norm):
                raise FloatingPointError(
                    "cloud gradient tensor %d has non-finite norm"
                    % tensor_index
                )
            squared_norms.append(tensor_norm * tensor_norm)

        gradient_norm = math.sqrt(math.fsum(squared_norms))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise FloatingPointError(
                "cloud gradient global L2 norm must be positive and finite, got %r"
                % gradient_norm
            )

        basis_direction = [tensor.to(self.device) / gradient_norm for tensor in gradient]
        unit_squared_norms = [
            tensor.detach().float().norm().item() ** 2
            for tensor in basis_direction
        ]
        unit_norm = math.sqrt(math.fsum(unit_squared_norms))
        if not math.isfinite(unit_norm) or not math.isclose(
                unit_norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise FloatingPointError(
                "normalized cloud direction must have global L2 norm 1, got %r"
                % unit_norm
            )
        if not all(torch.isfinite(tensor).all().item() for tensor in basis_direction):
            raise FloatingPointError("normalized cloud direction contains non-finite values")

        # Use a dedicated seed for the one-dimensional Rademacher coefficient.
        # Its sequence depends only on the experiment seed and round counter,
        # so paired cloud-data arms receive identical relative signs.
        direction_seed = int(self.args.manual_seed) + self.direction_counter
        direction_generator = torch.Generator(device="cpu")
        direction_generator.manual_seed(direction_seed)
        direction_bit = torch.randint(
            0, 2, (1,), generator=direction_generator, device="cpu"
        ).item()
        direction_sign = 1.0 if direction_bit else -1.0
        perturbation = [direction_sign * tensor for tensor in basis_direction]
        ggd_squared_norms = [
            tensor.detach().float().norm().item() ** 2
            for tensor in perturbation
        ]
        ggd_norm = math.sqrt(math.fsum(ggd_squared_norms))
        if not math.isfinite(ggd_norm) or ggd_norm <= 0.0:
            raise FloatingPointError(
                "sampled cloud direction must have positive finite global L2 norm"
            )

        logging.info(
            "[ZGR] cloud basis m=1: raw_L2=%.8g basis_L2=%.8g "
            "rademacher_seed=%d sign=%+.0f Vz_g_L2=%.8g",
            gradient_norm,
            unit_norm,
            direction_seed,
            direction_sign,
            ggd_norm,
        )
        self.direction_counter += 1
        return perturbation
