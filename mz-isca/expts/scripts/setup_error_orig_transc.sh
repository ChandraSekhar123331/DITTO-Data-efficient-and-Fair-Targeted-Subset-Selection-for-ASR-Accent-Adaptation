curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 



{

    HOME_PATH=$(pwd)
    QUARTZNET_SCRIPTS=$(cd ../../models/quartznet_asr; pwd)
    ERROR_MODEL_SCRIPTS=$(cd ../../models/error_model; pwd)

    accent=$1

    echo $accent

    cd $QUARTZNET_SCRIPTS
    . mz-isca-scripts/infer_transcriptions_on_seed_set.sh $accent
    cd $HOME_PATH
    cd $ERROR_MODEL_SCRIPTS
    . mz-isca-scripts/train_error_model_orig_transc.sh $accent
    cd $HOME_PATH

    echo 

} &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"