# declare -a speakers=('arabic' 'chinese' 'hindi' 'korean' 'spanish' 'vietnamese') 
# declare -a speakers=('canada' 'england' 'australia' 'us' 'philippines')
target=$4
budget=$3
feature=$2
accent=$1

echo "_____________________________________________________________"
echo "$accent"
# . run_random.sh $accent $feature $budget $target &> ./logs/random-log-"$accent"-"$feature"-"$budget"-"$target".txt
# echo "_____________________random completed____________________________"
. run.sh $accent $feature $budget $target &> ./logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt
echo "_________________________tss completed________________________"
# . run_within.sh $accent $feature $budget $target &> ./logs/within-log-"$accent"-"$feature"-"$budget"-"$target".txt
# echo "________________________within completed________________________"