homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=50
budget=100
feature="39"
# speaker=$1
speaker="ABA"
echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$speaker" "$feature"
echo
python TSS_within_equal_random.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature"
