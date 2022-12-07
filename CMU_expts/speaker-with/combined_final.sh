declare -a speakers=('ABA' 'ASI' 'BWC' 'HJK' 'NJS' 'PNV') 
for speaker in "${speakers[@]}"
do
	echo "__________________________"
	echo "$speaker"
	. run_random.sh $speaker
	. run.sh $speaker
#	. run_within_equal_random.sh $speaker
	echo "__________________________"
	echo
	echo
done

