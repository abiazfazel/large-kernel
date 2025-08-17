train
python train_mafunet_lk.py --data_root dataset --img_size 320 448 --epochs 280 --batch_size 6 --accum 4 --lr 2e-4 --weight_decay 2e-4 --amp --ema --pos_weight 2.0 --alpha 0.5 --lk_size 7 --lk_stages 2,3,4,5 --use_gcn --gcn_ks 11 --val_every 1 --threshold 0.5 --no_flops

test
python eval_test_mafunet_lk.py --data_root dataset --img_size 320 448 --batch_size 4 --ckpt runs/mafunet_lk/best.pth --threshold 0.5 --amp --tta --lk_size 7 --lk_stages 2,3,4,5 --use_gcn --gcn_ks 11 --save_csv runs/mafunet_lk/test_metrics.csv
