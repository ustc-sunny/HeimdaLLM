import logging
import os
from sre_parse import GLOBAL_FLAGS
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import post_complete_message_to_sweep_process, grad_aggregete

class FedSGDClientManager(ClientManager):
    def __init__(self, args, client, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.client = client
        self.num_rounds = args.comm_round
        self.round_idx = 0

        # 状态记录
        self.model_received = False
        self.perturbation_received = False
        logging.info("FedSGDClientManager __init__ called")

    def run(self):
        logging.info("FedSGDClientManager run called")
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT,
                                              self.handle_message_receive_aggregated_grad_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_MORE_V,
                                              self.calculate_more_v)
        ## new:
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEND_PERT_TO_CLIENT,
                                              self.handle_message_receive_pert_from_server)
        logging.info(f"Client finished registering handlers.")

    

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info("handle_message_init. client_index = " + str(client_index))
        # ad_hoc
        # self.trainer.trainer.model_trainer.cur_v_num_index += 1

        self.client.update_model(global_model_params)
        self.client.update_dataset(client_index)
        self.round_idx = 0
        # self.__train()
        self.data_id = 0
        self.train_with_data_id()

    def handle_message_receive_pert_from_server(self, msg_params):
        logging.info("handle_message_receive_pert_from_server")
        perturbation = msg_params.get(MyMessage.MSG_ARG_KEY_GRAD_PERT)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.client.trainer.client_trainer.set_perturbation(perturbation)

        self.perturbation_received = True
        self.try_start_training()

    # def start_training(self):
    #     self.round_idx = 0
    #     self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        # 方差足够小，清空暂存的fwdgrad
        if self.args.var_control:
            if self.args.perturbation_sampling:
                if self.data_id % 2:
                    self.client.trainer.client_trainer.old_grad = grad_aggregete(self.client.trainer.client_trainer.grad_pool)
                else:
                    self.client.trainer.client_trainer.old_grad = None
                self.client.trainer.client_trainer.grad_pool = []
            self.client.trainer.client_trainer.grad_for_var_check_list = []


        self.client.update_model(model_params)
        self.client.update_dataset(client_index)
        
        self.model_received = True
        
        self.try_start_training()
        # self.data_id = 0
        # self.train_with_data_id()

        # self.__train()

        

    def handle_message_receive_aggregated_grad_from_server(self, msg_params):
        logging.info("handle_message_receive_aggregated_grad_from_server")
        
        # 方差足够小，清空暂存的fwdgrad
        if self.args.var_control:
            if self.args.perturbation_sampling:
                if self.data_id % 2:
                    self.client.trainer.client_trainer.old_grad = grad_aggregete(self.client.trainer.client_trainer.grad_pool)
                else:
                    self.client.trainer.client_trainer.old_grad = None
                self.client.trainer.client_trainer.grad_pool = []
            self.client.trainer.client_trainer.grad_for_var_check_list = []

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.client.update_model(model_params)
        self.train_with_data_id()

    def train_with_data_id(self):
        logging.info("#######training########### round_id = %d data_id = %d" % (self.round_idx, self.data_id))
            
        weights, client_num = self.client.train_with_data_id(self.round_idx,self.data_id)
        if self.args.var_control:
            self.send_var_to_server(1,self.client.trainer.client_trainer.var)
            
        self.data_id += 1
        if self.data_id == len(self.client.train_local_list[0]):
            self.send_model_to_server(1, weights, client_num)
        else:
            self.send_grad_to_server(1, weights, client_num)

    
    
    def try_start_training(self):
        """当模型和扰动都收到后才开始训练"""
        if self.model_received and self.perturbation_received:
            logging.info("Both model and perturbation received. Start training.")
            self.round_idx += 1
            self.data_id = 0
            self.train_with_data_id()
            # self.__train()

            # 清空状态，准备下一轮
            self.model_received = False
            self.perturbation_received = False

            if self.round_idx == self.num_rounds - 1:
                post_complete_message_to_sweep_process(self.args)
                self.finish()

    # 方差太大，计算更多v
    def calculate_more_v(self,msg_params):
        self.data_id -= 1
        logging.info("calculate more v")
        self.train_with_data_id()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_grad_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_var_to_server(self, receive_id, var):
        logging.info("send_var_to_server")
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_VAR_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params("var", var)
        self.send_message(message)

    # def __train(self):
    #     logging.info("#######training########### round_id = %d" % self.round_idx)
    #     weights, client_num = self.trainer.train(self.round_idx)
    #     logging.info("start send gard to server")
    #     self.send_model_to_server(1, weights, client_num)#local_sample_num)

    

    