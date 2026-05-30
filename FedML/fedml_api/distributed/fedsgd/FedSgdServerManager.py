from distutils.ccompiler import gen_lib_options
import logging
import os, signal
import sys
import torch
import functorch

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager

class FedSGDServerManager(ServerManager):
    def __init__(self, args, server, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.server = server
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.pertubation = None
        logging.info("FedSGDServerManager __init__ called")

    def run(self):
        logging.info("FedSGDServerManager run called")
        super().run()

    def send_init_msg(self):
        # sampling clients
        self.client_indexes = self.server.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        
        global_model_params = self.server.get_global_model_params()
        
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        
        logging.info("client_indexes = " + str(self.client_indexes))
        logging.info("size = " + str(self.size))
        
        for process_id in range(2, self.size):
            self.send_message_init_config(process_id, global_model_params, self.client_indexes[process_id - 2])
            logging.info("process_id: " + str(process_id) + " client_index: " + str(self.client_indexes[process_id - 2]))

        self.send_message_init_config_to_cloud(0, global_model_params)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)
        
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER,
                                              self.aggregate_tmp_grad)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_VAR_TO_SERVER,
                                              self.get_var)
        ## new: 
        self.register_message_receive_handler(MyMessage.MSG_TYPE_CLOUD2S_SEND_PERT_TO_SERVER,
                                              self.handle_message_receive_pert_from_cloud)
        logging.info(f"Server finished registering handlers.")
    


    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        logging.info("handle_message_receive_model_from_client. sender_id = " + str(sender_id))

        self.server.add_local_trained_result(sender_id - 2, model_params, local_sample_number)
        b_all_received = self.server.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.server.aggregate(self.round_idx)
            # logging.info("retry_count: " + str(self.server.retry_count))
            if self.args.var_control and self.server.var > self.server.var_threthod and self.server.retry_count < self.server.max_retries:
                logging.info(f"current model is not good enough, calculate more v (retry {self.server.retry_count + 1}/{self.server.max_retries})")
                self.server.retry_count += 1
                for receiver_id in range(2, self.size):
                    self.send_message_cal_more_grad(receiver_id)
            else:
                self.server.retry_count = 0  # 重置重试计数
                self.server.test_on_server_for_all_clients(self.round_idx)

                # start the next round
                self.round_idx += 1
                if self.round_idx == self.round_num-1:
                    post_complete_message_to_sweep_process(self.args)
                    self.finish()
                    return
                if self.is_preprocessed:
                    if self.preprocessed_client_lists is None:
                        # sampling has already been done in data preprocessor
                        self.client_indexes = [self.round_idx] * self.args.client_num_per_round
                    else:
                        self.client_indexes = self.preprocessed_client_lists[self.round_idx]
                else:
                    # sampling clients
                    self.client_indexes = self.server.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                    self.args.client_num_per_round)
                

                for receiver_id in range(2, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                        self.client_indexes[receiver_id - 2])
                    
                # 将模型发给云端，用于bp训练
                self.send_message_aggregate_grad_to_cloud(0, global_model_params)
                
    def aggregate_tmp_grad(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.server.add_local_trained_result(sender_id - 2, model_params, local_sample_number)
        b_all_received = self.server.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.server.aggregate(self.round_idx)

            # logging.info("retry_count: " + str(self.server.retry_count))
            
            logging.info(f"var_control:" + str(self.args.var_control))
            logging.info(f"var:" + str(self.server.var))
            logging.info(f"var_threthod:" + str(self.server.var_threthod))
            logging.info(f"max_retries:" + str(self.server.max_retries))
            logging.info(f"retry_count:" + str(self.server.retry_count))

            if self.args.var_control and (self.server.var > self.server.var_threthod and self.server.retry_count < self.server.max_retries):
                logging.info(f"current model is not good enough, calculate more v (retry {self.server.retry_count + 1}/{self.server.max_retries})")
                self.server.retry_count += 1
                for receiver_id in range(2, self.size):
                    self.send_message_cal_more_grad(receiver_id)
            else:
                self.server.retry_count = 0  # 重置重试计数
                for receiver_id in range(2, self.size):
                    self.send_message_aggregate_grad_to_client(receiver_id, global_model_params,
                                                        self.client_indexes[receiver_id - 2])
   
    def handle_message_receive_pert_from_cloud(self, msg_params):
        logging.info("handle_message_receive_pert_from_cloud")
        perturbation = msg_params.get(MyMessage.MSG_ARG_KEY_GRAD_PERT)

        self.pertubation = perturbation
        logging.info("received perturbation from cloud")
        if self.pertubation is not None:
            for receiver_id in range(2, self.size):
                self.send_pert_to_client(receiver_id, self.pertubation,
                                self.client_indexes[receiver_id - 2])

    def get_var(self,msg_params):
        var = msg_params.get("var")
        self.server.var = var

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)
    
    def send_message_init_config_to_cloud(self, receive_id, global_model_params):
        message = Message(MyMessage.MSG_TYPE_S2CLOUD_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)

    def send_message_aggregate_grad_to_client(self, receive_id, global_model_params, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)


    def send_pert_to_client(self, receive_id, perturbation, client_index):
        logging.info("send_message_perturbation_to_server. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_PERT_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_PERT, perturbation)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)

    def send_message_cal_more_grad(self, receive_id):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_MORE_V, self.get_sender_id(), receive_id)
        self.send_message(message)

    
    ## 发送聚合后的模型到云端
    def send_message_aggregate_grad_to_cloud(self, receive_id, global_model_params):
        logging.info("send_message_aggregate_grad_to_cloud. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2CLOUD_SEND_GARD_TO_CLOUD, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)