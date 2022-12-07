homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
similarity="euclidean"
feature="39"
accents=$1
declare -a fxns=('FL' 'GC' 'LogD')
for fxn in "${fxns[@]}"
do
	echo
	echo "---------------------------------------------------------"
	echo "$fxn"
	echo "$target" "$budget" "$similarity" "$accents" "$fxn" "$feature"
	echo
#	python TSS_within.py --target "$target" --budget "$budget" --similarity "$similarity" --accent "$accents" --fxn "$fxn" --feature_type "$feature"
#	for file in $accents/manifests/TSS_output/within/budget_"$budget"/target_"$target"/"$fxn"/"$similarity"/"$feature"/run_*/t*/*;
#	do
#		cp $file ${file/train./dristi_train.}
#			sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' ${file/train./dristi_train.}
#	done
	cd "$finetunepath"
	. l2_accent_scripts/dristi_asr_finetune_within.sh
	. l2_accent_scripts/dristi_asr_test_within.sh
	cd "$homepath"
done
