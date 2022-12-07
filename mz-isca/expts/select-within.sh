declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us')
# declare -a accents=('indian' 'african' 'philippines' 'canada' 'england' 'australia' 'us' )
declare -a targets=('20')
declare -a budgets=('100' '200' '400' '500')

feature='w2v2'

for accent in "${accents[@]}"
do
	echo "__________________________"
	echo "$accent"
    for budget in "${budgets[@]}"
    do
        echo "$budget"
        for target in "${targets[@]}"
        do
            echo target "$target"
            python TSS_within.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature"
        done
        echo "____________________________________________________"
    done
    echo "$accent" completed
	echo
done


