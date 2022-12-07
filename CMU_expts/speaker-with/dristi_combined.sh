#declare -a speakers=('ABA' 'ASI' 'BWC' 'EBVS' 'ERMS' 'HJK' 'HKK' 'HQTV' 'LXC' 'MBMPS' 'NCC' 'NJS' 'PNV' 'RRBI' 'SKA' 'SVBI' 'THV' 'TLV' 'TNI' 'TXHC' 'YBAA' 'YDCK' 'YKWK' 'ZHAA')
#declare -a speakers=('ASI' 'BWC' 'EBVS' 'ERMS' 'HJK') 
declare -a speakers=('ERMS' 'HJK') 
for speaker in "${speakers[@]}"
do
	echo "__________________________"
	echo "$speaker"
#	. dristi_run_random.sh $speaker
	. dristi_run.sh $speaker
	echo "__________________________"
	echo
	echo
done

