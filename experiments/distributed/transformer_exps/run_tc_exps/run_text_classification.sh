client_num_per_round=$1
LR=$2
FL_ALG=$3
CLOUD_DATASET=${4:-}  # optional: cloud dataset for domain mismatch (e.g., yelp-p)

C_LR=0.01
S_LR=0.1
ROUND=1500
WORKER_NUM=1
model_type=distilbert
model_name=/home/zzkevin/models/distilbert-base-uncased

# model_type=bert
# model_name=bert-base-uncased
# model_type=bert
# model_name=bert-large-uncased
# model_type=albert
# model_name=/home/zzkevin/models/albert-base-v2
# model_type=roberta-large
# model_name=/home/zzkevin/models/roberta-large
# model_type=deberta
# model_name=microsoft/deberta-xlarge
train_batch_size=8
DATA_NAME=sst_2
# fold_name=${model_type}_${DATA_NAME}

if [ $model_type = "distilbert" ];then
  peft_method=adapter
else
  peft_method=bitfit
fi

PARTITION_METHOD="uniform_client_1000"
if [ $DATA_NAME = "agnews" ];then
  max_seq_length=64
  frequency_of_the_test=1
elif [ $DATA_NAME = "20news" ];then
  max_seq_length=256
  frequency_of_the_test=1
  #PARTITION_METHOD="uniform"
elif [ $DATA_NAME = "yelp-p" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "yahoo" ];then
  max_seq_length=256
  frequency_of_the_test=5
  PARTITION_METHOD="uniform_client_10000"
elif [ $DATA_NAME = "cnn_dailymail" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "cornell_movie_dialogue" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "ploner" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "semeval_2010_task8" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "sst_2" ];then
  max_seq_length=64
  frequency_of_the_test=1
else
  max_seq_length=256
  frequency_of_the_test=1
fi


LOG_FILE="fedavg_transformer_tc.log"
CI=0

DATA_DIR=../../../../fednlp_data

# Domain Mismatch: build cloud args if CLOUD_DATASET is specified
CLOUD_ARGS=""
if [ -n "$CLOUD_DATASET" ]; then
  # default cloud max_seq_length: look up per-dataset, fallback 256
  if [ "$CLOUD_DATASET" = "sst_2" ]; then
    CLOUD_SEQ=64
  elif [ "$CLOUD_DATASET" = "agnews" ]; then
    CLOUD_SEQ=64
  else
    CLOUD_SEQ=256
  fi
  CLOUD_ARGS="--cloud_dataset ${CLOUD_DATASET} --cloud_max_seq_length ${CLOUD_SEQ}"
fi

# 一个cloud进程，一个server进程，WORKER_NUM个client进程
PROCESS_NUM=`expr $WORKER_NUM + 2`
echo $PROCESS_NUM

hostname > mpi_host_file
if [ $FL_ALG = "FedAvg" ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --frequency_of_the_test $frequency_of_the_test \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    $CLOUD_ARGS \
    > ./log/new/fedavg_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}.log 2>&1
elif [ $FL_ALG = FedSgd ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --frequency_of_the_test $frequency_of_the_test \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    $CLOUD_ARGS \
    > ./log/new/fedsgd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_full.log 2>&1
else
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_cloud \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --frequency_of_the_test $frequency_of_the_test \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --eval_batch_size 8 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --worker_num $WORKER_NUM \
    --epochs 1 \
    --peft_method $peft_method \
    --forward_mode \
    --learning_rate $LR \
    --var_control \
    --perturbation_sampling \
    $CLOUD_ARGS \
    > ./log/new/test_fedFwd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_numerical.log 2>&1
# --perturbation_sampling \  --var_control \
fi