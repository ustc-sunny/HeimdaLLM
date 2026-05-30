from mpi4py import MPI

from .FedSgdServer import FedSGDServer
from .FedSgdClient import FedSGDClient
from .FedSgdCloud import FedSGDCloud

from .FedSgdClientManager import FedSGDClientManager
from .FedSgdServerManager import FedSGDServerManager
from .FedSgdCloudManager import FedSGDCloudManager


from ...standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from ...standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from ...standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
import logging

def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedSgd_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer=None, preprocessed_sampling_lists=None):
    # if process_id == 0:
    #     init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
    #                 test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
    #                 model_trainer, preprocessed_sampling_lists)
    # else:
    #     init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
    #                 train_data_local_dict, test_data_local_dict, model_trainer)
    if process_id == 1:
        logging.info("Server is initializing...")
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                    model_trainer, preprocessed_sampling_lists)
    elif process_id == 0:
        logging.info("Cloud is initializing...")
        init_cloud(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global, model_trainer)
    else:
        logging.info("Client is initializing...")
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict, test_data_local_dict, model_trainer)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, model_trainer, preprocessed_sampling_lists=None):
    # if model_trainer is None:
    #     if args.dataset == "stackoverflow_lr":
    #         model_trainer = MyModelTrainerTAG(model)
    #     elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
    #         model_trainer = MyModelTrainerNWP(model)
    #     else: # default model trainer is for classification problem
    #         model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 2
    server = FedSGDServer(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                  worker_num, device, args, model_trainer)

    # start the distributed training

    backend = args.backend
    if preprocessed_sampling_lists is None :
        server_manager = FedSGDServerManager(args, server, comm, rank, size, backend)
    else:
        server_manager = FedSGDServerManager(args, server, comm, rank, size, backend,
            is_preprocessed=True, 
            preprocessed_client_lists=preprocessed_sampling_lists)

    server_manager.send_init_msg()
    server_manager.run()



def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict,
                train_data_local_dict, test_data_local_dict, model_trainer=None):
    client_index = process_id - 2

    # if model_trainer is None:
    #     if args.dataset == "stackoverflow_lr":
    #         model_trainer = MyModelTrainerTAG(model)
    #     elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
    #         model_trainer = MyModelTrainerNWP(model)
    #     else: # default model trainer is for classification problem
    #         model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(client_index)
    backend = args.backend

    client = FedSGDClient(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                            train_data_num, device, args, model_trainer)

    client_manager = FedSGDClientManager(args, client, comm, process_id, size,backend)
    client_manager.run()

def init_cloud(args, device, comm, process_id, size, model, train_data_num, train_data_global, model_trainer):
    # if model_trainer is None:
    #     if args.dataset == "stackoverflow_lr":
    #         model_trainer = MyModelTrainerTAG(model)
    #     elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
    #         model_trainer = MyModelTrainerNWP(model)
    #     else: # default model trainer is for classification problem
    #         model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-2)
    backend = args.backend

    cloud = FedSGDCloud(train_data_global, train_data_num, device, args, model_trainer)
    # logging.info(str(cloud) + " initialized, about to create cloud manager")

    cloud_manager = FedSGDCloudManager(args, cloud, comm, process_id, size, backend)
    # logging.info(str(cloud_manager) + " initialized, about to run cloud manager")
    cloud_manager.run()