# declare -a accents=('indian' 'philippines' 'hongkong' 'scotland' 'african') 
declare -a accents=('hongkong' 'african') 
# declare -a accents=('indian' 'hongkong' 'philippines' 'scotland' 'malaysia')
target=$3
budget=$2
feature=$1

# declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
for accent in "${accents[@]}"
do
	echo "__________________________"
	echo "$accent"
	. run_without_random.sh $accent $feature $budget $target &> ./logs/random-log-"$accent"-"$feature"-"$budget"-"$target".txt
    echo "_____________________random completed____________________________"
# 	. run_without.sh $accent $feature $budget $target &> ./logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt
# 	echo "________________________tss completed________________________"
    echo "$accent" completed
	echo
done

