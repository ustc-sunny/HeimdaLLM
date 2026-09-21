from abc import ABC, abstractmethod
import hashlib
import h5py
import json

from data_preprocessing.base.base_data_loader import BaseDataLoader
from tqdm import tqdm
import logging
import numpy as np
import pickle
import os
import tempfile


class BaseDataManager(ABC):
    @abstractmethod
    def __init__(self, args, model_args, process_id, num_workers):
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.process_id = process_id
        self.num_workers = num_workers

        # TODO: add type comments for the below vars.
        self.train_dataset = None
        self.test_dataset = None
        self.train_examples = None
        self.test_examples = None
        self.train_loader = None
        self.test_loader = None
        self.client_index_list = None
        self.client_index_pointer = 0
        self.attributes = None

        self.partition_num = self.load_num_partitions(
            self.args.partition_file_path, self.args.partition_method)
        self.num_clients = int(self.partition_num * 0.8)
        # TODO: sync to the same logic to sample index
        # self.client_index_list = self.sample_client_index(process_id, num_workers)
        # self.client_index_list = self.get_all_clients()
        self.client_index_list = self.sample_client_index(process_id, num_workers)

    @staticmethod
    def load_attributes(data_path):
        data_file = h5py.File(data_path, "r", swmr=True)
        attributes = json.loads(data_file["attributes"][()])
        data_file.close()
        return attributes

    @staticmethod
    def load_num_clients(partition_file_path, partition_name):
        data_file = h5py.File(partition_file_path, "r", swmr=True)
        num_clients = int(data_file[partition_name]["n_clients"][()])
        data_file.close()
        return num_clients

    @staticmethod
    def load_num_partitions(partition_file_path, partition_name):
        data_file = h5py.File(partition_file_path, "r", swmr=True)
        partition_num = int(data_file[partition_name]["n_clients"][()])
        data_file.close()
        return partition_num

    @abstractmethod
    def read_instance_from_h5(self, data_file, index_list, desc):
        pass

    def sample_client_index(self, process_id, num_workers):
        '''
        Sample client indices according to process_id
        '''
        # process_id = 0 means this process is the server process
        if process_id == 0 or process_id == 1:
            return None
        else:
            return self._simulated_sampling(process_id)

    def _simulated_sampling(self, process_id):
        res_client_indexes = list()
        for round_idx in range(self.args.comm_round):
            if self.num_clients == self.num_workers:
                client_indexes = [client_index
                                  for client_index in range(self.num_clients)]
            else:
                nc = min(self.num_workers, self.num_clients)
                # make sure for each comparison, we are selecting the same clients each round
                np.random.seed(round_idx)
                client_indexes = np.random.choice(
                    range(self.num_clients),
                    nc, replace=False)
                # logging.info("client_indexes = %s" % str(client_indexes))
            res_client_indexes.append(client_indexes[process_id-1])
        return res_client_indexes

    def get_all_clients(self):
        return list(range(0, self.num_clients))

    def load_centralized_data(self, cut_off=None):
        state, res = self._load_data_loader_from_cache(-1)
        if state:
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(
                self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            for client_idx in tqdm(
                partition_file[partition_method]
                ["partition_data"].keys(),
                    desc="Loading index from h5 file."):
                train_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["train"][()][:cut_off])
                test_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["test"][()][:cut_off])
            train_data = self.read_instance_from_h5(data_file, train_index_list)
            test_data = self.read_instance_from_h5(data_file, test_index_list)
            data_file.close()
            partition_file.close()
            train_examples, train_features, train_dataset = self.preprocessor.transform(
                **train_data, index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(
                **test_data, index_list=test_index_list, evaluate=True)
            
            self._write_cache_atomically(
                res,
                (train_examples, train_features, train_dataset,
                 test_examples, test_features, test_dataset),
            )
        train_dl = BaseDataLoader(train_examples, train_features, train_dataset,
                              batch_size=self.train_batch_size,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)

        test_dl = BaseDataLoader(test_examples, test_features, test_dataset,
                             batch_size=self.eval_batch_size,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False)
        
        return train_dl, test_dl

    def load_federated_data(self, process_id, test_cut_off=None):
        if process_id == 0:
            return self._load_federated_data_server(test_cut_off=test_cut_off, test_only=False)
        else:
            return self._load_federated_data_local()
    
    def load_cloud_server_client_data(self, process_id, test_cut_off=None):
        if process_id == 1:
            return self._load_federated_data_server(test_cut_off=test_cut_off, test_only=True)
        elif process_id == 0:
            return self._load_federated_data_cloud()
        else:
            return self._load_federated_data_local()
        
    # def _load_federated_data_cloud(self):

    #     state, res = self._load_data_loader_from_cache(self.args.client_num_in_total)  # 用最后一个“client” id 来缓存 cloud data
    #     train_data_local_dict = None
    #     train_data_local_num_dict = None
    #     test_data_global = None
    #     test_data_local_dict = None
        
    #     if state:
    #         train_examples, train_features, train_dataset, _, _, _ = res
    #         logging.info("cloud train data size " + str(len(train_examples)))
    #     else:
    #         # 读取 HDF5 文件
    #         data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
    #         partition_file = h5py.File(self.args.partition_file_path, "r", swmr=True)
    #         partition_method = self.args.partition_method

    #         train_index_list = partition_file[partition_method][
    #                     "partition_data"][
    #                     str(self.args.client_num_in_total)]["train"][
    #                     ()]
            
    #         train_data = self.read_instance_from_h5(data_file, train_index_list)


    #         data_file.close()
    #         partition_file.close()

    #         train_examples, train_features, train_dataset = self.preprocessor.transform(
    #             **train_data, index_list=train_index_list)

    #         logging.info("cloud caching train data size " + str(len(train_examples)))

    #         with open(res, "wb") as handle:
    #             pickle.dump((train_examples, train_features, train_dataset), handle)

       
    #     train_data_global = BaseDataLoader(train_examples, train_features, train_dataset,
    #                                     batch_size=self.train_batch_size,
    #                                     num_workers=0,
    #                                     pin_memory=True,
    #                                     drop_last=False)
    #     train_data_num = len(train_examples)

    #     return (train_data_num, train_data_global, test_data_global,
    #             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)

    @staticmethod
    def _client_id_sort_key(client_id):
        try:
            return (0, int(client_id))
        except ValueError:
            return (1, client_id)

    def get_clients_indices(self, partition_file, partition_method, client_ids):
        """Merge train/test indices for an explicit list of partition clients."""
        partition_data = partition_file[partition_method]["partition_data"]
        train_index_list = []
        test_index_list = []
        for client_id in client_ids:
            client_id = str(client_id)
            if client_id not in partition_data:
                raise ValueError(
                    "client %s is not present in partition method %s"
                    % (client_id, partition_method)
                )
            train_index_list.extend(partition_data[client_id]["train"][()])
            test_index_list.extend(partition_data[client_id]["test"][()])
        return train_index_list, test_index_list

    def get_last_n_clients_indices(self, partition_file, partition_method, n):
        """
        Merge the last n clients declared by the partition file.

        The partition metadata is authoritative. In particular, do not use
        args.client_num_in_total here: it still contains the CLI default while
        the data manager is being constructed.
        """
        total_clients = int(partition_file[partition_method]["n_clients"][()])
        if n <= 0 or n > total_clients:
            raise ValueError(
                "cloud client count must be in [1, %d], got %d"
                % (total_clients, n)
            )
        client_ids = [str(client_id) for client_id in range(total_clients - n, total_clients)]
        return self.get_clients_indices(partition_file, partition_method, client_ids)

    def _resolve_cloud_client_ids(self, partition_file, partition_method):
        partition_data = partition_file[partition_method]["partition_data"]
        available_ids = sorted(
            (str(client_id) for client_id in partition_data.keys()),
            key=self._client_id_sort_key,
        )
        declared_num_clients = int(partition_file[partition_method]["n_clients"][()])
        if declared_num_clients != len(available_ids):
            raise ValueError(
                "partition method %s declares %d clients but contains %d client groups"
                % (partition_method, declared_num_clients, len(available_ids))
            )
        requested_ids = getattr(self.args, "cloud_client_ids", None)
        if requested_ids:
            client_ids = [client_id.strip() for client_id in requested_ids.split(",")]
            if any(not client_id for client_id in client_ids):
                raise ValueError("--cloud_client_ids contains an empty client ID")
            if len(set(client_ids)) != len(client_ids):
                raise ValueError("--cloud_client_ids contains duplicate client IDs")
            missing_ids = [client_id for client_id in client_ids if client_id not in partition_data]
            if missing_ids:
                raise ValueError(
                    "cloud client IDs are absent from partition method %s: %s"
                    % (partition_method, ",".join(missing_ids))
                )
            return client_ids

        client_count = getattr(self.args, "cloud_client_count", None)
        if client_count is None:
            return available_ids
        if client_count <= 0 or client_count > len(available_ids):
            raise ValueError(
                "cloud client count must be in [1, %d], got %d"
                % (len(available_ids), client_count)
            )
        return available_ids[-client_count:]
        
    def _load_federated_data_cloud(self):
        cache_id = "cloud"
        state, res = self._load_data_loader_from_cache(cache_id)
        # state, res = self._load_data_loader_from_cache(self.args.client_num_in_total)  # 用最后一个“client” id 来缓存 cloud data
        train_data_local_dict = None
        train_data_local_num_dict = None
        test_data_global = None
        test_data_local_dict = None
        
        if state:
            train_examples, train_features, train_dataset, _, _, _ = res
            logging.info("cloud train data size " + str(len(train_examples)))
        else:
            partition_method = self.args.partition_method
            with h5py.File(self.args.data_file_path, "r", swmr=True) as data_file, \
                    h5py.File(self.args.partition_file_path, "r", swmr=True) as partition_file:
                cloud_client_ids = self._resolve_cloud_client_ids(
                    partition_file, partition_method
                )
                train_index_list, _ = self.get_clients_indices(
                    partition_file, partition_method, cloud_client_ids
                )
                if not train_index_list:
                    raise ValueError("selected cloud clients contain no training examples")
                logging.info(
                    "cloud partition clients=%s train_index_size=%d",
                    cloud_client_ids,
                    len(train_index_list),
                )
                train_data = self.read_instance_from_h5(data_file, train_index_list)

            train_examples, train_features, train_dataset = self.preprocessor.transform(
                **train_data, index_list=train_index_list)
            #test_examples, test_features, test_dataset = self.preprocessor.transform(
             #   **test_data, index_list=test_index_list, evaluate=True)

            logging.info("cloud caching train data size " + str(len(train_examples)))

            test_examples, test_features, test_dataset = None, None, None

            # with open(res, "wb") as handle:
            #     pickle.dump((train_examples, train_features, train_dataset), handle)
            self._write_cache_atomically(
                res,
                (train_examples, train_features, train_dataset,
                 test_examples, test_features, test_dataset),
            )

       
        train_data_global = BaseDataLoader(train_examples, train_features, train_dataset,
                                        batch_size=self.train_batch_size,
                                        num_workers=0,
                                        pin_memory=True,
                                        drop_last=False)
        train_data_num = len(train_examples)

        return (train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)


    def _load_federated_data_server(self, test_only=True, test_cut_off=None):
        state, res = self._load_data_loader_from_cache(-1)
        train_data_local_dict = None
        train_data_local_num_dict = None
        test_data_local_dict = {}
        if state:
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
            logging.info("test data size "+ str(len(test_examples)))
            if train_dataset is None:
                train_data_num = 0
            else:
                train_data_num = len(train_dataset)
            logging.info("train data size "+ str(train_data_num))
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(
                self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            # test_examples = []
            # test_features = []
            # test_dataset = []
            for client_idx in tqdm(
                partition_file[partition_method]
                ["partition_data"].keys(),
                    desc="Loading index from h5 file."):
                train_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["train"][()])
                local_test_index_list = partition_file[partition_method][
                    "partition_data"][client_idx]["test"][()]
                test_index_list.extend(local_test_index_list)

            if not test_only:
                train_data = self.read_instance_from_h5(
                    data_file, train_index_list)
            if test_cut_off:
                test_index_list.sort()
            test_index_list = test_index_list[:test_cut_off]
            logging.info("caching test index size "+ str(len(test_index_list)) + "test cut off " + str(test_cut_off))

            test_data = self.read_instance_from_h5(data_file, test_index_list)

            data_file.close()
            partition_file.close()

            train_examples, train_features, train_dataset = None, None, None
            if not test_only:
                train_examples, train_features, train_dataset = self.preprocessor.transform(
                    **train_data, index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(
                **test_data, index_list=test_index_list)
            logging.info("caching test data size "+ str(len(test_examples)))

            self._write_cache_atomically(
                res,
                (train_examples, train_features, train_dataset,
                 test_examples, test_features, test_dataset),
            )

        if test_only or train_dataset is None:
            train_data_num = 0
            train_data_global = None
        else:
            train_data_global = BaseDataLoader(train_examples, train_features, train_dataset,
                                        batch_size=self.train_batch_size,
                                        num_workers=0,
                                        pin_memory=True,
                                        drop_last=False)
            train_data_num = len(train_examples)
            logging.info("train_dl_global number = " + str(len(train_data_global)))

        test_data_global = BaseDataLoader(test_examples, test_features, test_dataset,
                                      batch_size=self.eval_batch_size,
                                      num_workers=0,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False)

        logging.info("test_dl_global number = " + str(len(test_data_global)))


        return (train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)

    def _load_federated_data_local(self):
        partition_method = self.args.partition_method

        train_data_local_dict = {}
        test_data_local_dict = {}
        train_data_local_num_dict = {}
        # self.client_index_list = list(set(self.client_index_list))
        self.client_index_list = list(range(self.num_clients))
        logging.info("self.client_index_list = " + str(self.client_index_list))

        from multiprocessing import Process
        def add_local_dict(client_list):
            # h5py handles must be opened after fork; sharing the parent's
            # handles between cache workers can corrupt reads or deadlock.
            with h5py.File(self.args.data_file_path, "r", swmr=True) as worker_data_file, \
                    h5py.File(self.args.partition_file_path, "r", swmr=True) as worker_partition_file:
                for client_idx in client_list:
                    state, res = self._load_data_loader_from_cache(client_idx)
                    if state:
                        continue
                    train_index_list = worker_partition_file[partition_method][
                        "partition_data"][str(client_idx)]["train"][()]
                    test_index_list = worker_partition_file[partition_method][
                        "partition_data"][str(client_idx)]["test"][()]
                    train_data = self.read_instance_from_h5(
                        worker_data_file, train_index_list,
                        desc=" train data of client_id=%d [_load_federated_data_local] " % client_idx)
                    test_data = self.read_instance_from_h5(
                        worker_data_file, test_index_list,
                        desc=" test data of client_id=%d [_load_federated_data_local] " % client_idx)

                    train_examples, train_features, train_dataset = self.preprocessor.transform(
                        **train_data, index_list=train_index_list)
                    test_examples, test_features, test_dataset = self.preprocessor.transform(
                        **test_data, index_list=test_index_list, evaluate=True)

                    self._write_cache_atomically(
                        res,
                        (train_examples, train_features, train_dataset,
                         test_examples, test_features, test_dataset),
                    )

        def preprocess_client_caches(process_count):
            if process_count == 1:
                add_local_dict(range(self.num_clients))
                return

            process_list = []
            for process_index in range(process_count):
                start = int(process_index * (self.num_clients // process_count))
                end = int((process_index + 1) * (self.num_clients // process_count))
                if process_index == process_count - 1:
                    end = self.num_clients
                process = Process(target=add_local_dict, args=(range(start, end),))
                process_list.append(process)

            for process in process_list:
                process.start()
            for process in process_list:
                process.join()

            failed_processes = [
                (process.pid, process.exitcode)
                for process in process_list if process.exitcode != 0
            ]
            if failed_processes:
                raise RuntimeError(
                    "client cache preprocessing failed (pid, exitcode): %s"
                    % failed_processes
                )

        # 测试一下是否有cache的data，没有就多线程处理
        state, res = self._load_data_loader_from_cache(0)
        if not state:
            if self.num_clients <= 0:
                raise ValueError("local federated data contains no clients")
            cache_process_count = min(
                8,
                self.num_clients,
                max(1, int(self.num_workers)),
                max(1, os.cpu_count() or 1),
            )
            logging.info(
                "preprocessing client caches with %d process(es)",
                cache_process_count,
            )
            preprocess_client_caches(cache_process_count)

        with h5py.File(self.args.data_file_path, "r", swmr=True) as data_file, \
                h5py.File(self.args.partition_file_path, "r", swmr=True) as partition_file:
            for client_idx in self.client_index_list:
                state, res = self._load_data_loader_from_cache(client_idx)
                if state:
                    train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
                else:
                    train_index_list = partition_file[partition_method][
                        "partition_data"][str(client_idx)]["train"][()]
                    test_index_list = partition_file[partition_method][
                        "partition_data"][str(client_idx)]["test"][()]
                    train_data = self.read_instance_from_h5(
                        data_file, train_index_list,
                        desc=" train data of client_id=%d [_load_federated_data_local] " % client_idx)
                    test_data = self.read_instance_from_h5(
                        data_file, test_index_list,
                        desc=" test data of client_id=%d [_load_federated_data_local] " % client_idx)

                    train_examples, train_features, train_dataset = self.preprocessor.transform(
                        **train_data, index_list=train_index_list)
                    test_examples, test_features, test_dataset = self.preprocessor.transform(
                        **test_data, index_list=test_index_list, evaluate=True)

                    self._write_cache_atomically(
                        res,
                        (train_examples, train_features, train_dataset,
                         test_examples, test_features, test_dataset),
                    )

                train_loader = BaseDataLoader(
                    train_examples, train_features, train_dataset,
                    batch_size=self.train_batch_size,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                )
                train_data_local_dict[client_idx] = train_loader
                test_data_local_dict[client_idx] = -1
                train_data_local_num_dict[client_idx] = len(train_loader)

        logging.info(len(train_data_local_dict))
        logging.info(train_data_local_dict.keys())

        train_data_global, test_data_global, train_data_num = None, None, 0
        return (train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)
    

    @staticmethod
    def _write_cache_atomically(cache_path, payload):
        """Publish a complete pickle with one atomic rename."""
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="wb", dir=cache_dir,
                    prefix=os.path.basename(cache_path) + ".",
                    suffix=".tmp", delete=False) as handle:
                temp_path = handle.name
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            os.replace(temp_path, cache_path)
            temp_path = None
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _load_data_loader_from_cache(self, client_id):
        """
        Different clients has different cache file. client_id = -1 means loading the cached file on server end.
        """
        args = self.args
        model_args = self.model_args
        rank_cache_dir = os.path.join(
            model_args.cache_dir,
            "rank_%s" % getattr(self, "process_id", "unknown"),
        )
        os.makedirs(rank_cache_dir, exist_ok=True)
        source_identity = {}
        for source_name, source_path in (
            ("data", args.data_file_path),
            ("partition", args.partition_file_path),
        ):
            resolved_path = os.path.realpath(source_path)
            source_stat = os.stat(resolved_path)
            source_identity[source_name] = {
                "path": resolved_path,
                "device": source_stat.st_dev,
                "inode": source_stat.st_ino,
                "size": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
                "ctime_ns": source_stat.st_ctime_ns,
            }
        if str(client_id) == "cloud":
            source_identity["cloud_selection"] = {
                "client_ids": getattr(args, "cloud_client_ids", None),
                "client_count": getattr(args, "cloud_client_count", None),
            }
        source_fingerprint = hashlib.sha256(
            json.dumps(source_identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cached_features_file = os.path.join(
            rank_cache_dir, args.model_type + "_" + args.model_name.split("/")[-1] + "_cached_" + str(args.max_seq_length) + "_" + model_args.model_class + "_"
            + args.dataset + "_" + args.partition_method + "_" + str(client_id) + "_" + source_fingerprint
        )
        if os.path.exists(cached_features_file) and (
            (not model_args.reprocess_input_data and not model_args.no_cache)
            or (model_args.use_cached_eval_features and not model_args.no_cache)
        ):
            # logging.info(" Loading features from cached file %s", cached_features_file)
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = None, None, None, None, None, None
            with open(cached_features_file, "rb") as handle:
                train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = pickle.load(handle)
            return True, (train_examples, train_features, train_dataset, test_examples, test_features, test_dataset)
        return False, cached_features_file
