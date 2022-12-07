declare -a accents=('hongkong')
# declare -a accents=('indian' 'hongkong' 'philippines' 'scotland' 'malaysia')
target=$3
budget=$2
feature=$1

# declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
for accent in "${accents[@]}"
do
	echo "__________________________"
	echo "$accent"
	. run_random.sh $accent $feature $budget $target &> ./logs/random-log-"$accent"-"$feature"-"$budget"-"$target".txt
    echo "_____________________random completed____________________________"
	. run.sh $accent $feature $budget $target &> ./logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt
	echo "________________________tss completed________________________"
# 		. run_within.sh $accent $feature $budget $target &> ./logs/within-log-"$accent"-"$feature"-"$budget"-"$target".txt
# 	echo "________________________within completed________________________"
    echo "$accent" completed
	echo
done

