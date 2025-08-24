import logging
import os
from sre_parse import GLOBAL_FLAGS
import sys

import threading

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


## 接收
# MSG_TYPE_S2C_INIT_CONFIG
# MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT
# MSG_TYPE_C2C_SEND_PERT_TO_CLIENT


## 发送
# MSG_TYPE_C2S_SEND_GARD_TO_SERVER


class FedSGDClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

        # 状态记录
        self.model_received = False
        self.perturbation_received = False

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT,
                                              self.handle_message_receive_aggregated_grad_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEND_PERT_TO_CLIENT,
                                              self.handle_message_receive_pert_from_cloud)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info("handle_message_init. client_index = " + str(client_index))

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(client_index)
        self.round_idx = 0
        # self.__train()
        self.data_id = 0
        self.train_with_data_id()

    def handle_message_receive_pert_from_cloud(self, msg_params):
        logging.info("handle_message_receive_pert_from_cloud")
        perturbation = msg_params.get(MyMessage.MSG_ARG_KEY_GRAD_PERT)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.trainer.model_trainer.set_perturbation(perturbation)

        self.perturbation_received = True
        self.try_start_training()
        



    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(model_params)
        self.round_idx += 1

        self.model_received = True
        self.try_start_training()



    def try_start_training(self):
        """当模型和扰动都收到后才开始训练"""
        if self.model_received and self.perturbation_received:
            logging.info("Both model and perturbation received. Start training.")
            # 清空状态，准备下一轮
            self.model_received = False
            self.perturbation_received = False

            self.data_id = 0
            self.train_with_data_id()

            # 如果是最后一轮，发送完成消息
            if self.round_idx == self.num_rounds - 1:
                post_complete_message_to_sweep_process(self.args)
                self.finish()


    def send_gard_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GARD_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)



    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train(self.round_idx)
        self.send_gard_to_server(0, weights, local_sample_num)

    def train_with_data_id(self):
        for data_id in range(len(self.trainer.train_local_list[0])):
            logging.info("#######training########### round_id = %d data_id = %d" % (self.round_idx, data_id))

            weights, client_num = self.trainer.train_with_data_id(self.round_idx, data_id)
            self.trainer.update_model(weights)
            # self.trainer.update_dataset(client_index)
        
            self.data_id = data_id + 1
            
        if self.data_id == len(self.trainer.train_local_list[0]):
            self.send_gard_to_server(0, weights, client_num)
        
