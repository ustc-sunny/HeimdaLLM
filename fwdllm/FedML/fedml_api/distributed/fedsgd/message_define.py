class MyMessage(object):
    """
        message type definition
    """
    # cloud to client
    MSG_TYPE_C2C_SEND_PERT_TO_CLIENT = 5

    # server to cloud
    MSG_TYPE_S2CLOUD_INIT_CONFIG = 0
    MSG_TYPE_S2C_SEND_GARD_TO_CLOUD = 4
    
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SEND_GRAD_TO_CLIENT = 3

    # client to server
    MSG_TYPE_C2S_SEND_GRAD_TO_SERVER = 2
    

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_GRAD_PERT = "grad_pert"


    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"


