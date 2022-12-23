#!/bin/bash
#date
#now = $(date +"%I%M%S")
#echo $now
source activate /home/user/miniconda3/envs/tf3
cd /home/user/CV/Keras_insightface

echo "********************************************************************************************"
echo "***                                      WFH SWAP                                        ***"
echo "********************************************************************************************"

echo "********************"
echo "Download Model SWAP AN"
echo "********************"
python download_model.py --save_path_model_spoof existing_model/spoof_wfh.tflite --source_path_model_spoof sp_v000.tflite  --save_path_model_wfh existing_model/swap_wfh.tflite --source_path_model_wfh fr_wfh_v000.tflite --source_path_model_wfo fr_wfo_v000.tflite --save_path_model_wfo existing_model/swap_wfo.tflite
echo "*******************************"
echo "Fetching data and grouping data"
echo "*******************************"
date
date +"%I%M%S"
var=$(date +%I%M%S)
echo "$var"
echo "auto_train_no_mask/$var.h5"
python coba.py --dataset_temp train_set/swap_wfh_temp --dataset_base train_set/swap_wfh1/
echo "********************"
echo "Training Model"
echo "********************"
python train_nets.py --model checkpoints-no-mask/MobileFaceNet_SE_retrain.h5 --dataset train_set/swap_wfh1 --save_path auto_train_no_mask/swap_wfh/$var.h5 --epoch 1
echo "********************"
echo "Convert Model"
echo "********************"
python convert_tflite.py --model_path checkpoints/auto_train_no_mask/swap_wfh/$var --save_path checkpoints/auto_train_no_mask/swap_wfh --model_name $var
echo "********************"
echo "Clean Set"
echo "********************"
python clean_coba.py --dataset_temp train_set/swap_wfh_temp --dataset_base train_set/swap_wfh1/ --limit_dataset_base 10
echo "********************"
echo "Test and compare"
echo "********************"
python test_tflite.py --model checkpoints/auto_train_no_mask/swap_wfh/$var.tflite --dataset datasets/wings_data --pairs pairs/wings_pairs.txt --xlsx wings_data_$var.xlsx --tipe swap_wfh --model_existing existing_model/swap_wfh.tflite --upload_path fr_wfh_$var.tflite

echo "********************************************************************************************"
echo "***                                      WFO SWAP                                        ***"
echo "********************************************************************************************"

echo "*******************************"
echo "Fetching data and grouping data"
echo "*******************************"
date
date +"%I%M%S"
var=$(date +%I%M%S)
echo "$var"
echo "auto_train_no_mask/$var.h5"
python coba.py --dataset_temp train_set/swap_wfo_temp --dataset_base train_set/swap_wfo1/
echo "********************"
echo "Training Model"
echo "********************"
python train_nets.py --model checkpoints-no-mask/MobileFaceNet_SE_retrain.h5 --dataset train_set/swap_wfo1 --save_path auto_train_no_mask/swap_wfo/$var.h5 --epoch 10
echo "********************"
echo "Convert Model"
echo "********************"
python convert_tflite.py --model_path checkpoints/auto_train_no_mask/swap_wfo/$var --save_path checkpoints/auto_train_no_mask/swap_wfo --model_name $var
echo "********************"
echo "Clean Set"
echo "********************"
python clean_coba.py --dataset_temp train_set/swap_wfo_temp --dataset_base train_set/swap_wfo1/ --limit_dataset_base 10
echo "********************"
echo "Test and Compare Set"
echo "********************"
python test_tflite.py --model checkpoints/auto_train_no_mask/swap_wfo/$var.tflite --dataset datasets/wings_data_masked --pairs pairs/wings_pairs_mask.txt --xlsx wings_data_$var.xlsx --tipe swap_wfo --model_existing existing_model/swap_wfo.tflite --upload_path fr_wfo_$var.tflite

echo "********************************************************************************************"
echo "***                                      WFH SPOOF                                       ***"
echo "********************************************************************************************"

date
date +"%I%M%S"
var=$(date +%I%M%S)
echo "$var"

echo "********************"
echo "Training Model"
echo "********************"
python train_nets_spoof.py --data_dir antispoofing/train_set --epoch 1 --save_path checkpoints/auto_train_no_mask/spoof_wfh/$var --model_checkpoint antispoofing/model-best-v6-1.h5
echo "********************"
echo "Convert Model"
echo "********************"
python convert_tflite.py --model_path checkpoints/auto_train_no_mask/spoof_wfh/$var --save_path checkpoints/auto_train_no_mask/spoof_wfh --model_name $var
echo "********************"
echo "Test Model and Deploy"
echo "********************"
python test_tflite_spoof.py --model checkpoints/auto_train_no_mask/spoof_wfh/$var.tflite --existing_model existing_model/spoof_wfh.tflite --test_dir datasets/spoof_test --upload_path sp_$var.tflite

echo "********************"
echo "ALL DONE"
echo "********************"
