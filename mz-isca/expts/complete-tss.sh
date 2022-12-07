# declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
# declare -a accents=('indian' 'hongkong' 'philippines' 'scotland' 'malaysia')

# declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'scotland' 'ireland' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
# declare -a accents=('indian' 'scotland' 'hongkong' 'malaysia' 'bermuda' 'southatlandtic' 'wales') 
# declare -a accents=('indian' 'hongkong' 'malaysia') 
declare -a accents=('australia' 'canada') 
declare -a targets=('20')
# declare -a budgets=('200' '100' '300' '500' '700')
declare -a budgets=('200')
# declare -a budgets=('100' '400')
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
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
            
            python TSS_random.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature" >> 'tss-logs.txt' 2>&1
            echo "_____random completed______"
            
            python TSS_within.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature" >> 'tss-logs.txt' 2>&1
            echo "_____within completed______"
            
            for fxn in "${fxns[@]}"
            do
                echo "$fxn"
                python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accent" --fxn "$fxn" --feature_type "$feature" >> 'tss-logs.txt' 2>&1
            done
            echo "________tss completed______"
        done
        echo "____________"$budget"_done___________________________"
    done
    echo "$accent" completed
	echo "___"
done


