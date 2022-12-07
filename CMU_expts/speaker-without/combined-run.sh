declare -a speakers=('ABA' 'BWC' 'HJK')
# declare -a speakers=('ABA' 'ASI' 'BWC' 'HJK' 'EBVS' 'NJS' 'PNV')
# declare -a speakers=('HJK')
target=$3
budget=$2
feature=$1

# declare -a speakers=('african' 'philippines')
# declare -a speakers=('canada' 'england' 'australia' 'us' 'philippines' 'indian' 'african')
for speaker in "${speakers[@]}"
do
	echo "__________TSS for________________"
	echo "$speaker"
# 	. run.sh $speaker $feature $budget $target &> ./logs/tss-log-"$speaker"-"$feature"-"$budget"-"$target".txt
    . selection.sh $speaker $feature $budget $target &> ./logs/selection-log-"$speaker"-"$feature"-"$budget"-"$target".txt
	echo "__________________________"
#     echo "$speaker" within completed
	echo
done