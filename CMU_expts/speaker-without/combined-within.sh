declare -a speakers=('ABA' 'ASI' 'BWC' 'HJK' 'NJS' 'PNV')
# declare -a speakers=('ABA' 'ASI' 'BWC' 'HJK' 'EBVS' 'NJS' 'PNV')
# declare -a speakers=('HJK')
target=$3
budget=$2
feature=$1

# declare -a speakers=('african' 'philippines')
# declare -a speakers=('canada' 'england' 'australia' 'us' 'philippines' 'indian' 'african')
for speaker in "${speakers[@]}"
do
	echo "__________within for________________"
	echo "$speaker"
	. run_within_equal_random.sh $speaker $feature $budget $target &> ./logs/within-log-"$speaker"-"$feature"-"$budget"-"$target".txt
	echo "__________________________"
    echo "$speaker" within completed
	echo
done