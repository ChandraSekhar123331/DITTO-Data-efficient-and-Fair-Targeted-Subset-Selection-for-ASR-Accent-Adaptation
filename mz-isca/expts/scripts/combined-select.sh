declare -a accents=('indian')
# ('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'bermuda' 'southatlandtic' 'wales' 'malaysia')

target=20
feature='wv10_100'
b1=2500
b2=250

for accent in "${accents[@]}"
do
#     . scripts/tss.sh $accent $b1 $target $feature
	echo "______________________________________________________________________________"
	echo "$accent"
	. scripts/err_select.sh $accent $target $b1 $b2 $feature
	echo "____________________________________________________"
	echo
done