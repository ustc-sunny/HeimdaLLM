import logging
import os
from sre_parse import GLOBAL_FLAGS
import sys

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
# MSG_TYPE_S2CLOUD_INIT_CONFIG
# MSG_TYPE_S2CLOUD_SEND_GARD_TO_CLOUD

## 发送
# MSG_TYPE_C2C_SEND_PERT_TO_SERVER

class FedSGDCloudManager(ClientManager):
    def __init__(self, args, cloud, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.cloud = cloud
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.model_pool = []

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2CLOUD_SEND_GARD_TO_CLOUD,
                                              self.handle_message_receive_aggregated_grad_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2CLOUD_INIT_CONFIG,
                                              self.handle_message_init)
        logging.info(f"Cloud finished registering handlers.")
        
    
    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)

        logging.info("handle_message_init.")

        self.cloud.update_model(global_model_params)
        self.round_idx = 0
        self.train_bp()
        logging.info("finish train_bp")
        self.create_perturbation_and_send_to_server()


    # 接收服务端聚合的梯度
    def handle_message_receive_aggregated_grad_from_server(self, msg_params):
        logging.info("handle_message_receive_aggregated_grad_from_server")

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)

        self.cloud.update_model(model_params)
        logging.info("update model")

        self.train_bp()
        self.create_perturbation_and_send_to_server()

    def send_pert_to_server(self, receive_id, perturbation):
        logging.info("send_message_perturbation_to_server. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_CLOUD2S_SEND_PERT_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_PERT, perturbation)
        self.send_message(message)


    def train_bp(self):
        logging.info("####### cloud training ########### round_id = %d" % self.round_idx)
        self.cloud.train_model_bp()
        self.round_idx += 1

        if self.round_idx == self.num_rounds - 2:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def create_perturbation_and_send_to_server(self):
        logging.info("create_perturbation_and_send_to_server")

        perturbation = self.cloud.create_perturbation()
        self.send_pert_to_server(1, perturbation)
        logging.info("finish send perturbation to server")
            
        