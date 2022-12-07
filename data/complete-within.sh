declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 
declare -a targets=('20')
declare -a budgets=('200')
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
# feature='w2v2'
feature='39'
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
            
            python TSS_random.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature"
            echo "_____random completed______"
            python TSS_within_random.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature"
            echo "_____within completed______"
            for fxn in "${fxns[@]}"
            do
                echo "$fxn"
                python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accent" --fxn "$fxn" --feature_type "$feature"
            done
            echo "________tss completed______"
        done
        echo "____________________________________________________"
    done
    echo "$accent" completed
	echo "___"
done


