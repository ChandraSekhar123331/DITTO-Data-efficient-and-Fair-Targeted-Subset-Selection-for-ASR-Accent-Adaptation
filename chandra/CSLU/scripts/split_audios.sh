curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name


python VAD/Run_VAD.py &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
