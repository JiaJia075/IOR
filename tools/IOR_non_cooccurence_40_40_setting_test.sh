#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=3

config_path="configs/ior/ior_non_cooccurence_40_40_setting_each_class_10"
config_name="gfl_r50_fpn_1x_coco_incre_41-80_cats_unrepeated.py"


model_save_path="results/ior_non_cooccurence_40_40_setting_each_class_10"
model_save_name="ior_coco_incre_41-80_cats"
model_test_name="ior_coco_incre_41-80_cats_test"

# train 1-40
python tools/test.py --config "$config_path/$config_name"\
                    --checkpoint "$model_save_path/$model_save_name/epoch_12.pth"\
                    --work-dir "$model_save_path/$model_test_name"
                    --out "$model_save_path/$model_test_name/results.pkl"
