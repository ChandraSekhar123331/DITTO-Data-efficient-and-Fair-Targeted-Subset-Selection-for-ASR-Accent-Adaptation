declare -a speakers=('ABA' 'ASI' 'BWC' 'HJK' 'NJS' 'PNV' 'EBVS' 'ERMS')
# declare -a speakers=('NJS' 'PNV')
target=$3
budget=$2
feature=$1

# declare -a speakers=('african' 'philippines')
# declare -a speakers=('canada' 'england' 'australia' 'us' 'philippines' 'indian' 'african')
for speaker in "${speakers[@]}"
do
	echo "__________________________"
	echo "$speaker"
	. run_random.sh $speaker $feature $budget $target &> ./logs/random-log-"$speaker"-"$feature"-"$budget"-"$target".txt
	. run.sh $speaker $feature $budget $target &> ./logs/tss-log-"$speaker"-"$feature"-"$budget"-"$target".txt
	. run_within_equal_random.sh $speaker $feature $budget $target &> ./logs/within-log-"$speaker"-"$feature"-"$budget"-"$target".txt
	echo "__________________________"
    echo "$speaker" completed
	echo
done