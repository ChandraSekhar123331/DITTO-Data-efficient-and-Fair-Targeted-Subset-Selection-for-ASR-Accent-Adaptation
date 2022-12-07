# declare -a accents=("assamese_female_english" "manipuri_female_english" "kannada_male_english" "malayalam_male_english" "tamil_male_english" "rajasthani_female_english" "gujarati_female_english" "hindi_male_english")
declare -a accents=("malayalam_male_english" "manipuri_female_english" "rajasthani_male_english" "kannada_male_english" "assamese_female_english" )
declare -a methods=("FL2MI" "GCMI" "LogDMI")
declare -a budgets=(800)
declare -a targets=(10)
declare -a runs=(1 2 3)
for run in "${runs[@]}"; do
    for budget in "${budgets[@]}"; do
        for target in "${targets[@]}"; do
            for method in "${methods[@]}"; do
                for accent in "${accents[@]}"; do
                    eta=1.0
                    features=39
                    random_budget=100
                    bash random_on_SMI_indic.sh $accent $budget $target $method $eta 39 $run $random_budget
                    echo
                    echo
                    bash finetune_random_on_SMI_indic.sh $accent $budget $target $method $eta 39 $run $random_budget
                done
            done
        done
    done
done

for run in "${runs[@]}"; do
    for budget in "${budgets[@]}"; do
        for target in "${targets[@]}"; do
            for method in "${methods[@]}"; do
                for accent in "${accents[@]}"; do
                    eta=1.0
                    features=39
                    random_budget=200
                    bash random_on_SMI_indic.sh $accent $budget $target $method $eta 39 $run $random_budget
                    echo
                    echo
                    bash finetune_random_on_SMI_indic.sh $accent $budget $target $method $eta 39 $run $random_budget
                done
            done
        done
    done
done

