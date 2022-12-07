# declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
# declare -a accents=('indian' 'hongkong' 'philippines' 'scotland' 'malaysia')

# declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'scotland' 'ireland' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
# declare -a accents=('indian' 'scotland' 'hongkong' 'malaysia' 'bermuda' 'southatlandtic' 'wales') 
declare -a accents=('scotland' 'african' 'philippines' 'indian' 'hongkong')  
declare -a targets=('20')
# declare -a budgets=('100' '200' '500')
declare -a budgets=('200' '500')
# '100' '300' '700')
# declare -a budgets=('100' '400')
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
# feature='w2v2'
feature='wv10_100'
eta=1.0
similarity="euclidean"

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
            
            python TSS_without_random.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature" >> '39-logs.txt' 2>&1
            echo "_____without random completed______"
            
            
            for fxn in "${fxns[@]}"
            do
                echo "$fxn"
                python TSS_without.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accent" --fxn "$fxn" --feature_type "$feature" >> '39-logs.txt' 2>&1
            done
            echo "________tss completed______"
        done
        echo "____________________________________________________"
    done
    echo "$accent" completed
	echo "___"
done


