declare -a accents=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese')

feature='39'
b1=3500
b2=250


for accent in "${accents[@]}"
do
	echo "______________________________________________________________________________"
	echo "$accent"
#     . scripts/tss.sh $accent $b1 $target $feature > tss-logs.txt
	# . scripts/ts_top.sh $accent $target $b1 $b2 $feature
	# . scripts/ts_rand.sh $accent $target $b1 $b2 $feature
	. scripts/ts_err.sh $accent $target $b1 $b2 $feature
    # . scripts/random.sh $accent $b2
	echo "____________________________________________________"
	echo
	echo
done


