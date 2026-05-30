#!/usr/bin/env python3
"""
为数据集生成 uniform_client_1000 分区
基于 test.ipynb 中的方法
"""
import h5py
import numpy as np
import json
import argparse


def generate_uniform_partition(data_file, partition_file, client_num=1000, seed=42):
    """
    为数据集生成uniform分区
    
    Args:
        data_file: 数据h5文件路径
        partition_file: 分区h5文件路径
        client_num: 客户端数量，默认1000
        seed: 随机种子
    """
    np.random.seed(seed)
    
    print(f"Reading data from {data_file}")
    
    # 读取数据文件获取索引列表
    with h5py.File(data_file, "r") as f:
        attributes = json.loads(f["attributes"][()])
        
        # 获取训练集和测试集索引
        if "train_index_list" in attributes:
            train_index_list = attributes["train_index_list"]
            test_index_list = attributes["test_index_list"]
        else:
            # 如果没有预分割，按9:1分割
            total_index_list = attributes["index_list"]
            train_length = int(len(total_index_list) * 0.9)
            train_index_list = total_index_list[:train_length]
            test_index_list = total_index_list[train_length:]
    
    print(f"Train samples: {len(train_index_list)}")
    print(f"Test samples: {len(test_index_list)}")
    
    # 转换为numpy数组并随机打乱
    train_indices = np.array(train_index_list, dtype=np.int64)
    test_indices = np.array(test_index_list, dtype=np.int64)
    
    train_indices = np.random.permutation(train_indices)
    test_indices = np.random.permutation(test_indices)
    
    # 确保train数据能被client_num整除（丢弃多余的）
    train_size = (len(train_indices) // client_num) * client_num
    train_indices = train_indices[:train_size]
    
    # 分割成client_num份
    partition_train = np.array_split(train_indices, client_num)
    partition_test = np.array_split(test_indices, client_num)
    
    print(f"Each client gets ~{len(partition_train[0])} train samples")
    print(f"Each client gets ~{len(partition_test[0])} test samples")
    
    # 写入分区文件
    print(f"Writing partition to {partition_file}")
    partition_name = f"uniform_client_{client_num}"
    
    with h5py.File(partition_file, "a") as f:
        # 如果已存在，先删除
        if partition_name in f:
            print(f"Warning: {partition_name} already exists, deleting...")
            del f[partition_name]
        
        # 写入客户端数量
        f[f"{partition_name}/n_clients"] = client_num
        
        # 写入每个客户端的数据
        for i in range(client_num):
            train_path = f"{partition_name}/partition_data/{i}/train"
            test_path = f"{partition_name}/partition_data/{i}/test"
            f[train_path] = partition_train[i]
            f[test_path] = partition_test[i]
            
            if i % 100 == 0:
                print(f"  Written client {i}/{client_num}")
    
    print(f"✅ Successfully generated {partition_name} partition!")
    
    # 验证
    with h5py.File(partition_file, "r") as f:
        n_clients = f[f"{partition_name}/n_clients"][()]
        print(f"\nVerification: n_clients = {n_clients}")
        sample_train = f[f"{partition_name}/partition_data/0/train"][()]
        sample_test = f[f"{partition_name}/partition_data/0/test"][()]
        print(f"Client 0: {len(sample_train)} train, {len(sample_test)} test samples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate uniform_client_1000 partition for datasets"
    )
    
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to data h5 file (e.g., fednlp_data/data_files/sst_2_data.h5)"
    )
    
    parser.add_argument(
        "--partition_file",
        type=str,
        required=True,
        help="Path to partition h5 file (e.g., fednlp_data/partition_files/sst_2_partition.h5)"
    )
    
    parser.add_argument(
        "--client_num",
        type=int,
        default=1000,
        help="Number of clients (default: 1000)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    generate_uniform_partition(
        args.data_file,
        args.partition_file,
        args.client_num,
        args.seed
    )


if __name__ == "__main__":
    main()
