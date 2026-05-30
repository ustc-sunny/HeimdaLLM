import logging

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedTransformerTrainer(ModelTrainer):

    def __init__(self, client_trainer, server_trainer, cloud_trainer, model):
        super().__init__(model)
        # self.model_trainer = trainer
        self.client_trainer = client_trainer
        self.server_trainer = server_trainer
        self.cloud_trainer = cloud_trainer
        self.model = model

    def get_model_params(self):
        return None

    def set_model_params(self, params):
        pass

    # def get_model_params(self):
    #     return self.model.cpu().state_dict()
    
    # def get_model(self):
    #     return self.model

    # def set_model_params(self, model_parameters):
    #     self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Client(%d)" % self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.client_trainer.train_dl = train_data
        self.client_trainer.train_model(device=device, args=args)

    def train_bp(self, train_data, device, args):
        logging.info("Cloud(%d)" % self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.cloud_trainer.train_dl = train_data
        self.cloud_trainer.train_model_bp(device=device)
        logging.info("Cloud(%d)" % self.id + ":| Train Model BP Done")

    def test(self, test_data, device, args=None):
        pass

    def test_on_the_server(self, device, args=None):
        self.server_trainer.eval_model(device=device)
        return True

    # 在前test_num个batch上进行快速的test
    def quick_test(self, device, test_num=1000):
        acc = self.model_trainer.quick_test(device=device,test_num=test_num)
        return acc
    
    def quick_test_loss(self, device, test_num=1000):
        loss = self.model_trainer.quick_test_loss(device=device,test_num=test_num)
        return loss