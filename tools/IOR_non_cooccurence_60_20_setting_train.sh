#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=6

coco_dataset_path="./dataset/coco"
dataset_image_folder_name="train2017"

config_path="configs/ior/ior_non_cooccurence_60_20_setting_each_class_10"
config_name_task1="gfl_r50_fpn_1x_coco_first_60_cats.py"
config_name_task2="gfl_r50_fpn_1x_coco_incre_61-80_cats_unrepeated.py"

generated_numImages_per_class=10

generated_path="./results_generated/ior_non_cooccurence_60_20_setting"
generated_folder_name1="first_60_cats_each_class_$generated_numImages_per_class"

agumented_replay_output_name="augemented_replay_60_20_setting"
ori_annotation_json_name="instances_train2017_ior_sel_last_20_cats_non_cooccurence"

model_save_path="results/ior_non_cooccurence_60_20_setting_each_class_10"
model_save_name_task1="ior_coco_first_60_cats"
model_save_name_task2="ior_coco_incre_61-80_cats"

# train 1-60
python tools/train.py --config "$config_path/$config_name_task1"\
                                                        --work-dir "$model_save_path/$model_save_name_task1"

# increment 61-80
python tools/inversion/singlebox_dataset_with_statistics_coco.py --numImages_per_class $generated_numImages_per_class \
                                                                                   --numClasses 60 \
                                                                                   --outdir "$generated_path/sampled_bbox/$generated_folder_name1"
python tools/inversion/main_gfl_generate_statictis_coco.py --save_path "$generated_path/generated_sample/$generated_folder_name1" \
                                                            --sample_path "$generated_path/sampled_bbox/$generated_folder_name1" \
                                                            --config "$config_path/$config_name_task1" \
                                                            --inversion_checkpoint "$model_save_path/$model_save_name_task1/epoch_12.pth"\
                                                            --bs 60
python tools/inversion/agumented_replay.py --anno_file "$coco_dataset_path/annotations/${ori_annotation_json_name}.json" \
                                                      --sampled_annotation_path "$generated_path/generated_sample/$generated_folder_name1" \
                                                      --img_path "$coco_dataset_path/$dataset_image_folder_name" \
                                                      --agumented_dataset_img_path "$coco_dataset_path/train2017_${agumented_replay_output_name}" \
                                                      --agumented_dataset_json_path "$coco_dataset_path/annotations_with_generates/instances_train2017_${agumented_replay_output_name}.json"
python tools/train.py --config "$config_path/$config_name_task2"\
                                                        --work-dir "$model_save_path/$model_save_name_task2"

echo "所有脚本都已运行完毕。"