accent=$1
budget=$2
target=$3
SMI_method=$4
eta=$5
features=$6
run=$7
random_budget=$8

base_path=/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/
PRETRAINED_CKPTS=$base_path/models/pretrained_checkpoints/
results_prefix_path=$base_path/data/$accent/manifests/TSS_output/all/budget_$budget/target_$target/$SMI_method/eta_$eta/euclidean/$features/run_$run/
echo Finetuning for accent=$accent budget=$budget $SMI_method run=$run
model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/ind_"$accent"/all/budget_"$budget"/target_"$target"/"$SMI_method"/eta_"$eta"/euclidean/"$features"/run_"$run"/random/
echo making $model_dir before finetuning
echo
mkdir -p $model_dir

# echo 
# echo **Warning:Chandra : This finetuning doesnot save the model for each epoch.
# echo **It only saves the best model. That too will be cleared after the testing ends.**
# echo **Kindly notice**
# echo
python3 -u $base_path/models/quartznet_asr/finetune.py \
--batch_size=16 \
--num_epochs=100 \
--eval_freq=1 \
--train_freq=30 \
--lr=1e-5 \
--wav_dir="" \
--train_manifest=$results_prefix_path/random/selected-$random_budget.json \
--val_manifest=$base_path/data/$accent/manifests/dev.json \
--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
--output_dir=$model_dir/recent \
--best_dir=$model_dir/best \
--early_stop_patience=10 \
--zero_infinity \
--save_after_each_epoch \
--turn_bn_eval \
--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
--lr_decay=warmup \
--seed=42 \
--optimizer=novograd \
> $model_dir/train_log.txt


echo Testing for accent=$accent budget=$budget $SMI_method run=$run
echo
# model_dir variable is already created above.
python3 -u $base_path/models/quartznet_asr/inference.py \
--batch_size=64 \
--output_file=$model_dir/test_out.txt \
--wav_dir="" \
--val_manifest=$base_path/data/$accent/manifests/test.json \
--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
--ckpt=$model_dir/best/Jasper.pt \
> $model_dir/test_infer_log.txt

output_dir=$results_prefix_path/random-output-$random_budget/
mkdir -p $output_dir
cp $model_dir/test_infer_log.txt $output_dir

# rm -r $model_dir