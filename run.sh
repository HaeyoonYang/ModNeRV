# train
python train.py --opt_path cfgs/uvg-12.5M.yaml --dataset Beauty --epochs 300
python train.py --opt_path cfgs/bunny-0.75M.yaml --dataset bunny --epochs 300

# # eval
# python eval.py --test_folder experiments/6.5M/Beauty --visualize

## train - compression
# python compress_train.py --compress_folder experiments/6.5M/Beauty --prune_epochs 50 --quant_epochs 30

## eval - compression
# python compress_eval.py --test_folder experiments/6.5M/Beauty --quant_bit 8 --visualize
