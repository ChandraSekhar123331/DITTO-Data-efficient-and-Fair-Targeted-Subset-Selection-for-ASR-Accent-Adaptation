#declare -a speakers=('ABA' 'ASI' 'BWC' 'EBVS' 'ERMS' 'HJK' 'HKK' 'HQTV' 'LXC' 'MBMPS' 'NCC' 'NJS' 'PNV' 'RRBI' 'SKA' 'SVBI' 'THV' 'TLV' 'TNI' 'TXHC' 'YBAA' 'YDCK' 'YKWK' 'ZHAA')
declare -a speakers=('THV' 'TLV' 'TNI' 'TXHC' 'YBAA' 'YDCK' 'YKWK' 'ZHAA')
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

