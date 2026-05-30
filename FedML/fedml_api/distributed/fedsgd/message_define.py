class MyMessage(object):
    """
        message type definition
    """

    # cloud to server (new)
    MSG_TYPE_CLOUD2S_SEND_PERT_TO_SERVER = 10

    # server to cloud (new)
    MSG_TYPE_S2CLOUD_INIT_CONFIG = 0
    MSG_TYPE_S2CLOUD_SEND_GARD_TO_CLOUD = 4

    # server to client (new)
    MSG_TYPE_S2C_SEND_PERT_TO_CLIENT = 9
    
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2
    MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT = 5
    MSG_TYPE_S2C_MORE_V = 7

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4
    MSG_TYPE_C2S_SEND_GRAD_TO_SERVER = 6
    MSG_TYPE_C2S_SEND_VAR_TO_SERVER = 8

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"


    """
        message payload keywords definition
    """
    MSG_ARG_KEY_GRAD_PERT = "grad_pert"
    MSG_ARG_KEY_CLIENT_INDEXES = "client_indexes"

    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"


