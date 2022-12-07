homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
similarity="euclidean"
eta=1.0
feature="39"
accents=$1
#declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
fxns=$2
for budget in $3
#for budget in 200 300 400 600 800
do 
	for fxn in "${fxns[@]}"
	do
		echo
		echo "---------------------------------------------------------"
		echo "$fxn"
		echo "$target" "$budget" "$similarity" "$eta" "$accents" "$fxn" "$feature"
		echo
		python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accents" --fxn "$fxn" --feature_type "$feature"
		for file in $accents/manifests/TSS_output/all/budget_"$budget"/target_"$target"/"$fxn"/eta_"$eta"/"$similarity"/"$feature"/run_*/t*/*;
		do
			cp $file ${file/train./dristi_train.}
			sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' ${file/train./dristi_train.}
		done
		cd "$finetunepath"
		. l2_accent_scripts/dristi_asr_finetune.sh
		. l2_accent_scripts/dristi_asr_test.sh
		cd "$homepath"
	done
done
