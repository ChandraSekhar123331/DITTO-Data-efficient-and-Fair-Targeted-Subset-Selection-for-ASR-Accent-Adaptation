declare -a accents=('assamese_female_english' 'gujarati_female_english' 'manipuri_female_english' 'hindi_male_english' 'rajasthani_male_english' 'tamil_male_english' 'kannada_male_english' 'malayalam_male_english') 

# for budget in 100 200
for budget in 200
do
    for accent in "${accents[@]}"
    do
       echo
       echo "-----------------testing random selections--------------------"
       echo "$budget"
       for run in 1 2 3
       do
           file_dir=random/$budget/run_$run
           mkdir -p file_dir
           cd "$finetunepath"
           echo "---------------beginning testing----------------------------"
           . scripts/test.sh $accent $file_dir
           cd "$homepath"
       done
   done
done