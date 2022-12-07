homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=10
similarity="euclidean"
eta=1.0
feature="39"
accents=$1
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
for budget in 200 300 400 800
do 
	for fxn in "${fxns[@]}"
	do
		echo
		echo "---------------------------------------------------------"
		echo "$fxn"
		echo "$target" "$budget" "$similarity" "$eta" "$accents" "$fxn" "$feature"
		echo
		python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accents" --fxn "$fxn" --feature_type "$feature"
#		cd "$finetunepath"
#		. l2_accent_scripts/asr_finetune.sh
#		. l2_accent_scripts/asr_test.sh
#		cd "$homepath"
	done
done
