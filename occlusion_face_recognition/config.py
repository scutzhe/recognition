import torch

configurations = {
    1: dict(
        SEED=1337,  # random seed for reproduce results

        # DATA_ROOT='/home/shengyang/haige_dataset/face_dataset2',
        DATA_ROOT='/home/shengyang/haige_dataset/face_occusion',
        # the parent root where your train/val/test data are stored
        MODEL_ROOT='./real_occlusion2_model',  # the root to buffer your checkpoints
        LOG_ROOT='./real_occlusion2_log',  # the root to log your train/val status

        BACKBONE_RESUME_ROOT='./real_occlusion_model/Backbone_FAN_IR_SE_101_Epoch_38_Batch_344318_Time_2019-11-01-09-08_checkpoint.pth',
        # BACKBONE_RESUME_ROOT='./model/Backbone_FAN_IR_SE_101_Epoch_10_Batch_90620_Time_2019-10-13-18-47_checkpoint.pth',
        # BACKBONE_RESUME_ROOT='./model/Backbone_IR_SE_101_Epoch_24_Batch_314592_Time_2019-10-01-13-49_checkpoint.pth',
        # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT='./real_occlusion_model/Head_ArcFace_Epoch_38_Batch_344318_Time_2019-11-01-09-08_checkpoint.pth',
        # HEAD_RESUME_ROOT='./model/Head_ArcFace_Epoch_1_Batch_9062_Time_2019-10-14-05-56_checkpoint.pth',
        # HEAD_RESUME_ROOT='./model/Head_ArcFace_Epoch_10_Batch_90620_Time_2019-10-13-18-47_checkpoint.pth',
        # HEAD_RESUME_ROOT='./model/Head_ArcFace_Epoch_24_Batch_314592_Time_2019-10-01-13-49_checkpoint.pth',
        # the root to resume training from a saved checkpoint

        BACKBONE_NAME='FAN_IR_SE_101',
        # BACKBONE_NAME='IR_SE_101',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50'
        # 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

        # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        HEAD_NAME='ArcFace',
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        # BATCH_SIZE=16,
        BATCH_SIZE=960,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        # LR=0.0005,  # initial LR
        LR=0.0001,  # initial LR
        NUM_EPOCH=30,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
        MOMENTUM=0.9,
        STAGES=[10, 20, 25],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=True,
        # flag to use multiple GPUs; if you choose to train with single GPU,
        # you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        # GPU_ID=[0, 1, 2, 3, 4],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=32,
    ),
}
