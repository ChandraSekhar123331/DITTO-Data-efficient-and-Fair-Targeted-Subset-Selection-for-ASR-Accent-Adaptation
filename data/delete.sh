homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=50
budget=100
similarity="euclidean"
eta=1.0
accents=$1
declare -a fxns=('FL1MI' 'FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
    echo
    echo
    echo "*********************************************"
    echo $homepath
    echo $finetunepath
    echo "*********************************************"
    echo
    echo 
    cd "$finetunepath"
    . scripts/delete.sh
    cd "$homepath"
done
