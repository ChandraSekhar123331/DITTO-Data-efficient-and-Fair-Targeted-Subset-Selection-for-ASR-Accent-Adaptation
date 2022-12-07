accent=$1
_dir=$2
for run in 1 2 3
do
	echo running index "$run"
	python finetune_w2v2-libri_indic_training.py --accent "$accent" --_dir "$_dir" --run "$run"
done
