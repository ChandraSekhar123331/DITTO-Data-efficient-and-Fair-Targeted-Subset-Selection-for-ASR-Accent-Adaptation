DATA=$(
  cd ../../data
  pwd
)
PRETRAINED_CKPTS=$(
  cd ../pretrained_checkpoints
  pwd
)
out_json_path=$base/error_model/
for seed in 1 2 3; do
  echo $accent seed $seed
  python3 -u error_model_sampling.py \
    --selection_json_file=$DATA/$accent/selection.json \
    --seed_json_file=$DATA/$accent/seed.json \
    --error_model_weights=$PRETRAINED_CKPTS/error_models/$accent/seed_"$seed"/best/weights.pkl \
    --random_json_path=$DATA/$accent/manifests/train/random \
    --output_json_path=$DATA/$accent/manifests/train/error_model \
    --exp_id=$seed
done
