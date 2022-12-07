homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
similarity="euclidean"
feature="39"
speaker=$1
declare -a fxns=('FL' 'GC' 'LogD') 
for fxn in "${fxns[@]}"
do
	echo
	echo "---------------------------------------------------------"
	echo "$fxn"
	echo "$target" "$budget" "$similarity" "$speaker" "$fxn" "$feature"
	echo
#	python TSS_within.py --target "$target" --budget "$budget" --similarity "$similarity" --speaker "$speaker" --fxn "$fxn" --feature_type "$feature"
#	for file in $speaker/manifests/TSS_output/within/budget_"$budget"/target_"$target"/"$fxn"/"$similarity"/"$feature"/run_*/t*/*;
#	do
#		cp $file ${file/train./dristi_train.}
#			sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' ${file/train./dristi_train.}
#	done
	cd "$finetunepath"
	. l2_speaker-W_scripts/asr_finetune_within.sh
	. l2_speaker-W_scripts/asr_test_within.sh
	cd "$homepath"
done
