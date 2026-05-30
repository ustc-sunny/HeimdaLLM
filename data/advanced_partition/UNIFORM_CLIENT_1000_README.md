# 为新数据集生成 uniform_client_1000 分区

## 📋 问题背景

新加入的数据集（20news, cnn_dailymail, ploner, semeval_2010_task8, sst_2）的partition文件中没有 `uniform_client_1000` 分区方法，导致运行时出现：
```
KeyError: "Unable to open object (object 'uniform_client_1000' doesn't exist)"
```

## ✅ 解决方案

### 方法一：使用自动化脚本（推荐）

已创建自动化脚本为所有新数据集生成 `uniform_client_1000` 分区：

```bash
cd /home/zzkevin/HeimdaLLM/xiexiu/data/advanced_partition
./generate_all_uniform_1000.sh
```

**执行结果：**
- ✅ 20news - 成功（1000客户端，每客户端约11条训练数据）
- ❌ cnn_dailymail - 失败（数据文件损坏）
- ✅ ploner - 成功（1000客户端，每客户端约14条训练数据）
- ✅ semeval_2010_task8 - 成功（1000客户端，每客户端约8条训练数据）
- ✅ sst_2 - 成功（1000客户端，每客户端约6条训练数据）

### 方法二：单独为某个数据集生成分区

```bash
cd /home/zzkevin/HeimdaLLM/xiexiu

python -m data.advanced_partition.generate_uniform_client_1000 \
    --data_file fednlp_data/data_files/sst_2_data.h5 \
    --partition_file fednlp_data/partition_files/sst_2_partition.h5 \
    --client_num 1000 \
    --seed 42
```

**参数说明：**
- `--data_file`: 数据h5文件路径
- `--partition_file`: 分区h5文件路径
- `--client_num`: 客户端数量（默认1000）
- `--seed`: 随机种子（默认42）

## 📊 生成的分区结构

在partition文件中会创建如下结构：
```
uniform_client_1000/
├── n_clients = 1000
└── partition_data/
    ├── 0/
    │   ├── train (数组)
    │   └── test (数组)
    ├── 1/
    │   ├── train
    │   └── test
    ...
    └── 999/
        ├── train
        └── test
```

## 🔍 验证分区是否成功

```python
import h5py

with h5py.File('fednlp_data/partition_files/sst_2_partition.h5', 'r') as f:
    print('Available partitions:')
    for key in f.keys():
        print(f'  - {key}')
    
    # 检查uniform_client_1000
    if 'uniform_client_1000' in f:
        n_clients = f['uniform_client_1000/n_clients'][()]
        train_0 = f['uniform_client_1000/partition_data/0/train'][()]
        test_0 = f['uniform_client_1000/partition_data/0/test'][()]
        print(f'\n✅ uniform_client_1000 exists!')
        print(f'   Clients: {n_clients}')
        print(f'   Client 0: {len(train_0)} train, {len(test_0)} test')
```

## 📝 技术实现

基于 `test.ipynb` 中的方法：

1. **读取数据索引**：从data文件的attributes中读取train_index_list和test_index_list
2. **随机打乱**：使用numpy.random.permutation随机打乱索引
3. **均匀分割**：使用numpy.array_split将数据均匀分配给1000个客户端
4. **写入h5文件**：按照标准格式写入partition文件

## ⚠️ 注意事项

1. **数据量问题**：某些数据集（如sst_2）数据量较小，1000个客户端会导致每个客户端只有很少的数据（6-8条）
2. **客户端数量调整**：如果数据量小，可以考虑使用更少的客户端（如100或30），只需修改`--client_num`参数
3. **cnn_dailymail问题**：该数据文件似乎已损坏，需要重新下载

## 🔧 自定义客户端数量

如果想生成其他数量的客户端（如100），只需修改参数：

```bash
python -m data.advanced_partition.generate_uniform_client_1000 \
    --data_file fednlp_data/data_files/sst_2_data.h5 \
    --partition_file fednlp_data/partition_files/sst_2_partition.h5 \
    --client_num 100 \  # 修改为100
    --seed 42
```

这会在partition文件中创建 `uniform_client_100` 分区。

## 📚 相关文件

- `generate_uniform_client_1000.py` - 单个数据集分区生成脚本
- `generate_all_uniform_1000.sh` - 批量处理所有新数据集的shell脚本
- `test.ipynb` - 原始实现参考（Cell 18-23）
