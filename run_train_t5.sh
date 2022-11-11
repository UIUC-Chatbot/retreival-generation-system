python finetune.py \
    --model_name "t5" \
    --dataset_path "/raid/projects/wentaoy4/save_dataset" \
    --model_path "google/flan-t5-large" \
    --device "cuda:0" \
    --logger_path "/raid/projects/wentaoy4/log/t5_finetune_b128_e10_lr5e06.log" \
    --saved_model_path "/raid/projects/wentaoy4/model_weight/t5_finetune_b128_e10_lr5e06.pt" \
    --batch_size 1 \
    --outer_batch_size 128 \
    --epochs 10 \
    --lr 5e-6