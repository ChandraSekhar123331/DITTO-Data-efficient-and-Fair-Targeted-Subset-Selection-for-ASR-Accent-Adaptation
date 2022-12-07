accent=$1
budget=$2
target=$3
SMI_method=$4
eta=$5
features=$6
run=$7
entropy_budget=$8

echo generating entropy-samples for accent=$accent budget=$budget $SMI_method run=$run

base_path=/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization
results_prefix_path=$base_path/data/$accent/manifests/TSS_output/all/budget_$budget/target_$target/$SMI_method/eta_$eta/euclidean/$features/run_$run
train_json=$results_prefix_path/train/train.json
echo train.json is $train_json
echo
echo Creating directory $results_prefix_path/random/
echo
mkdir -p $results_prefix_path/random/