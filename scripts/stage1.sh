source ./scripts/config.sh
accelerate launch train_svd_tracklet.py \
    --pretrained_model_name_or_path="<YOUR_stable-video-diffusion-img2vid_WEIGHT_FOLDER_PATH>" \
    --per_gpu_batch_size=2 --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --checkpointing_steps=10000 --checkpoints_total_limit=3 \
    --learning_rate=3e-5 --lr_warmup_steps=750 \
    --seed=1 \
    --mixed_precision="fp16" \
    --validation_steps=1000 \
    --train_mode="stage1" \
    --pretrain_unet="YOUR_presvd+precam_WEIGHT_FOLDER_PATH" 

