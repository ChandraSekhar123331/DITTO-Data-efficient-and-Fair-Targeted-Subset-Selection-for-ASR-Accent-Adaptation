homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=$4
budget=$3
feature=$2
accent=$1
echo
echo "-----------------------within----------------------------------"
echo "$target" "$budget" "$accent" "$feature"
echo
# python TSS_within.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature"
# african/manifests/TSS_output/within/budget_200/target_20/self/run_1/train
for file in $accent/manifests/TSS_output/within/budget_"$budget"/target_"$target"/self/run_*/t*/train*;
do
    cp $file ${file/train./dristi_train.}
    sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points\/jan_19/g' ${file/train./dristi_train.}
done
echo "---------------TSS done, beginning finetuning----------------------------"
cd "$finetunepath"
. mz_accent_scripts/dristi_asr_finetune_within.sh >> "$homepath"/logs/within-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
echo "---------------beginning testing----------------------------"
. mz_accent_scripts/dristi_asr_test_within.sh >> "$homepath"/logs/within-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
cd "$homepath"