homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
eta=1.0
similarity="euclidean"
target=$4
budget=$3
feature=$2
accent=$1
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
# declare -a fxns=('GCMI')
for fxn in "${fxns[@]}"
do
    echo
#     echo "------------------------TSS---------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$eta" "$accent" "$fxn" "$feature"
    echo
#    python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accent" --fxn "$fxn" --feature_type "$feature"
	for file in $accent/manifests/TSS_output/all/budget_"$budget"/target_"$target"/"$fxn"/eta_"$eta"/"$similarity"/"$feature"/run_*/t*/train*;
	do
		cp $file ${file/train./dristi_train.}
		sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points\/jan_19/g' ${file/train./dristi_train.}
	done
    cd "$finetunepath"
    . mz_accent_scripts/dristi_asr_finetune.sh >> "$homepath"/logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
    . mz_accent_scripts/dristi_asr_test.sh >> "$homepath"/logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
    cd "$homepath"
done
