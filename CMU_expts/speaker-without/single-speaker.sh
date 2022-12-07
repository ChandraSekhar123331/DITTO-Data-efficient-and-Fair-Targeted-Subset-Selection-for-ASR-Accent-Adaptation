target=$4
budget=$3
feature=$2
speaker=$1
# EBVS
echo "_____________________________________________________________"
echo "$speaker"
# . run_random.sh $speaker $feature $budget $target &> ./logs/random-log-"$speaker"-"$feature"-"$budget"-"$target".txt
echo "_____________________random completed____________________________"
. run.sh $speaker $feature $budget $target &> ./logs/tss-log-"$speaker"-"$feature"-"$budget"-"$target".txt
echo "________________________tss completed________________________"
. run_within_equal_random.sh $speaker $feature $budget $target &> ./logs/within-log-"$speaker"-"$feature"-"$budget"-"$target".txt
echo "______________________within completed_________________________"