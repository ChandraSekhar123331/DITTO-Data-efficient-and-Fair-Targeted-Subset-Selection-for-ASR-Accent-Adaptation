accent=$1
# declare -a accents=('indian' 'hongkong' 'philippines' 'scotland' 'malaysia' 'african')
target=$4
budget=$3
feature=$2


echo "__________________________"
echo "$accent"
. dristi_run_random.sh $accent $feature $budget $target &> ./logs/random-log-"$accent"-"$feature"-"$budget"-"$target".txt
echo "_____________________random completed____________________________"
. dristi_run.sh $accent $feature $budget $target &> ./logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt
echo "________________________tss completed________________________"
# 		. dristi_run_within.sh $accent $feature $budget $target &> ./logs/within-log-"$accent"-"$feature"-"$budget"-"$target".txt
# 	echo "________________________within completed________________________"
echo "$accent" completed
echo

