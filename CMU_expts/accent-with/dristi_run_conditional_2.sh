homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
similarity="euclidean"
eta=1.0
feature="39"
accents=$1
other_accent=$2
declare -a fxns=('FLMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
	echo
	echo "---------------------------------------------------------"
	echo "$fxn"
	echo "$target" "$budget" "$similarity" "$eta" "$accents" "$fxn" "$feature"
	echo
	#    python TSS_conditional.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accents" --other_accent "$other_accent" --fxn "$fxn" --feature_type "$feature"
	for file in $accents/manifests/TSS_output/all/budget_"$budget"/target_"$target"/conditional_"$fxn"_"$other_accent"/eta_"$eta"/"$similarity"/"$feature"/run_*/t*/*;
	do
		cp $file ${file/train./dristi_train.}
		sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' ${file/train./dristi_train.}
	done
	cd "$finetunepath"
	. l2_accent_scripts/dristi_asr_finetune_conditional.sh
	. l2_accent_scripts/dristi_asr_test_conditional.sh
	cd "$homepath"
done
