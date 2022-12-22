device='0'
# train 5fold
save_dir='../save_dir/train_fake_bfo/'

for fold_idx in {0..4}
do
    echo === fold $fold_idx ===
    # train 49
    python3 train.py \
        -c=config_trainfake.json \
        -d=$device \
        --fold_idx=$fold_idx \
        --save_dir=$save_dir/fold$fold_idx \
        --seed=42
    python3 test_1fold.py \
        -c=$save_dir/fold0/config.json \
        -f=$fold_idx \
        -s=fold${fold_idx}_test \
        -o=$save_dir
done
python3 test.py \
    --config=$save_dir/fold0/config.json \
    --output_dir=$save_dir \
    --device=$device\
    --suffix=5fold_ensemble\