declare -a speakers=('ABA' 'ASI' 'BWC' 'EBVS' 'ERMS' 'HJK' 'HKK' 'HQTV' 'LXC' 'MBMPS' 'NCC' 'NJS' 'PNV' 'RRBI' 'SKA' 'SVBI' 'THV' 'TLV' 'TNI' 'TXHC' 'YBAA' 'YDCK' 'YKWK' 'ZHAA')
#declare -a speakers=('ABA' 'ASI' 'BWC' 'EBVS' 'ERMS') 
target=10
budget=100
feature="TRILL"
for speaker in "${speakers[@]}"
do
	echo "__________________________"
	echo "$speaker"
	. run_random_x.sh $speaker
	. run_x.sh $speaker
	. run_within_equal_random_x.sh $speaker
	echo "__________________________"
	echo
	echo
done

