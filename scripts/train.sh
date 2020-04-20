#!/usr/bin/env bash

DATE=`date '+%m-%d-%H-%M'`
output_dir="outputs/flickr_ResNet50_pascal"
#pretrain_dir="outputs/vg_detection"
model_name="e2e_flickr_bottom_up_faster_rcnn_R_101_FPN_1x"
train_instance="ddpnResNet50_softmax_lr_0p1_reg_0p5.hidden_1024_diverse.sent_graph_top10_visualGraph_two_stage_rel_sample2"

gpu_num=1
RANDOM=1001
batch_num=24
batch_num_test=1
export CUDA_VISIBLE_DEVICES=1

# multi GPU command backup
# MODEL.ROI_BOX_HEAD.POOLER_SCALES $(0.0625,)
## (0.25, 0.125, 0.0625, 0.03125)
## "BottomUpMLPFeatureExtractor", "BottomUpFPN2MLPFeatureExtractor", "FPN2MLPFeatureExtractor", "BottomUpTopFeatureExtractor", 'FeatureExtractor'
## "Bottom-Up-R-101-C3", "Bottom-Up-R-101-C4", "Bottom-Up-R-101-C4-FPN", "R-101-FPN", "Bottom-Up-R-101-C4-Top"
## language embedding ("EASY", "SKIPTH", "GRU")
## CUDA_LAUNCH_BLOCKING=1 debug the cuda device-side assert
## SOLVER.IMS_PER_BATCH $[$gpu_num*batch_per_gpu], TEST.IMS_PER_BATCH $[$gpu_num*batch_per_gpu_test]
## EF: early fusion. transform([vis_feat, mesh-grid]), LF: later fusion.  transform(vis_feat), meshgrid
#              --test-while-training \

python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=$RANDOM \
              tools/train_net.py --config-file "./configs/$model_name.yaml" \
              --skip-test --test-while-training \
              OUTPUT_DIR "$output_dir/ddpnResNet50_softmax_lr_0p1_reg_0p5.hidden_1024_diverse.sent_graph_top10_visualGraph_two_stage_rel_sample2" \
              SOLVER.IMS_PER_BATCH $batch_num \
              SOLVER.TYPE "SGD" \
              SOLVER.BASE_LR 0.1 \
              SOLVER.REGLOSS_FACTOR 0.1 \
              SOLVER.RELATION_FACTOR 1.0 \
              MODEL.VG.PHRASE_SELECT_TYPE 'Mean' \
              MODEL.VG.PHRASE_EMBED_TYPE 'BaseSent' \
              MODEL.VG.TWO_STAGE 'True' \
              MODEL.VG.TOPN 10 \
              MODEL.VG.JOINT_TRANS 'True' \
              MODEL.VG.SPATIAL_FEAT 'True' \
              MODEL.RELATION_ON 'True' \
              MODEL.RELATION.RELATION_FEATURES 'True' \
              MODEL.RELATION.INTRA_LAN 'True' \
              MODEL.RELATION.INCOR_ENTITIES_IN_RELATION 'True' \
              MODEL.RELATION.INTRA_LAN_PASSING_TIME 1 \
              MODEL.RELATION.VISUAL_GRAPH 'True' \
              MODEL.RELATION.VISUAL_GRAPH_PASSING_TIME 1 \
              MODEL.RELATION.USE_RELATION_CONST 'True' \
              MODEL.RELATION.REL_CONST_TYPE 'Softmax' \
              MODEL.RELATION.REL_PAIR_IOU 'True' \
              MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 14 \
              DATALOADER.NUM_WORKERS 8 \
              SOLVER.CHECKPOINT_PERIOD 4000 \
              SOLVER.START_SAVE_CHECKPOINT 4000 \
              SOLVER.MAX_ITER 120001 \
              SOLVER.STEPS "(20000, 40000)" \
              TEST.IMS_PER_BATCH $batch_num_test \
              MODEL.WEIGHT "$output_dir/$train_instance/checkpoints/model_0044000.pth" \
              MODEL.USE_DET_PRETRAIN ""

#MODEL.USE_DET_PRETRAIN "$output_dir/ddpn_softmax_lr_0p05_reg_0p5.elmo_sent_mean.hidden_1024_diverse.top10_baseline/checkpoints/model_0044000.pth"
# MODEL.WEIGHT "$output_dir/$train_instance/checkpoints/model_0004000.pth"