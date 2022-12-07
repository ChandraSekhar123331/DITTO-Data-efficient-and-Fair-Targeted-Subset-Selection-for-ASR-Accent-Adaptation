accent=$1
budget=$2
target=$3
SMI_method=$4
eta=$5
features=$6
run=$7
entropy_budget=$8

base_path=/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/
PRETRAINED_CKPTS=$base_path/models/pretrained_checkpoints/
results_prefix_path=$base_path/mz-expts/$accent/manifests/TSS_output/all/budget_$budget/target_$target/$SMI_method/eta_$eta/euclidean/$features/run_$run/


echo $accent $run $budget
echo
echo
model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/mcv-accent_"$accent"/all/budget_"$budget"/target_"$target"/"$SMI_method"/eta_"$eta"/euclidean/"$features"/run_"$run"/entropy/
python3 -u $base_path/models/quartznet_asr/inference.py \
--batch_size=64 \
--output_file=$model_dir/test_out.txt \
--wav_dir="" \
--val_manifest=$base_path/mz-expts/$accent/manifests/test.json \
--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
--ckpt=$model_dir/best/Jasper.pt \
> $model_dir/test_infer_log.txt

output_dir=$results_prefix_path/entropy-output/
echo making dir $output_dir
mkdir -p $output_dir
cp $model_dir/test_infer_log.txt $output_dir
#     done
# done
# rm -r $PRETRAINED_CKPTS/quartznet/finetuned/mcv-accent_"$accent"/all/budget_$budget/target_"$target"