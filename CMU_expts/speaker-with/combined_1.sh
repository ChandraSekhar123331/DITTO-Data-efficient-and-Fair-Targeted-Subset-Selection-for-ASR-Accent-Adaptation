#declare -a speakers=('ABA' 'ASI' 'BWC' 'EBVS' 'ERMS' 'HJK' 'HKK' 'HQTV' 'LXC' 'MBMPS' 'NCC' 'NJS' 'PNV' 'RRBI' 'SKA' 'SVBI' 'THV' 'TLV' 'TNI' 'TXHC' 'YBAA' 'YDCK' 'YKWK' 'ZHAA')
declare -a speakers=('HJK' 'HKK' 'HQTV' 'LXC' 'MBMPS' 'NCC')
for speaker in "${speakers[@]}"
do
	echo "__________________________"
	echo "$speaker"
	. run_random_0.sh $speaker
	. run_0.sh $speaker
	. run_within_equal_random_0.sh $speaker
	echo "__________________________"
	echo
	echo
done

