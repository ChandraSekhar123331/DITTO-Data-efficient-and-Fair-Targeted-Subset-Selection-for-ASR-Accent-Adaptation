# declare -a accents=('african' 'australia' 'bermuda' 'canada' 'england' 'hongkong' 'indian' 'ireland' 'malaysia' 'philippines' 'scotland' 'southatlandtic' 'us' 'wales')
# declare -a accents=('african' 'australia' 'canada' 'england' 'hongkong' 'indian' 'ireland' 'philippines' 'scotland' 'southatlandtic' 'us')
# removed malaysia, bermuda, wales as their selection file is empty

# target=20
# feature='wv10_100'
# b1=2000
b2=250
accent=$1


# for accent in "${accents[@]}"
# do
echo "_______________________doing setup for error model_______________________________"
echo "$accent"
scripts/setup_error_orig_transc.sh $accent
echo "done with accent = $accent"
# done



echo "___running error_model sampling and finetuning__"
echo $accent
scripts/ts_err_pure_orig_transc.sh $accent $b2

# for accent in "${accents[@]}"
# do
# 	echo "______________________________________________________________________________"
# 	echo "$accent"
#     # scripts/tss.sh $accent $b1 $target $feature
# 	# scripts/ts_err.sh $accent $target $b1 $b2 $feature
# 	# scripts/ts_top.sh $accent $target $b1 $b2 $feature
# 	# scripts/ts_rand.sh $accent $target $b1 $b2 $feature # this should be random within the SMI
#     # scripts/random.sh $accent $b2 # This is global random
# 	echo "____________________________________________________"
# 	echo
# 	echo
# done