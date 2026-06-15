set -e  # stop if any training fails

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "========== Starting pi0 =========="
lerobot-train --dataset.repo_id=training/fighter_V1.63 --policy.type=pi0 --output_dir=outputs/train/pi0_fighter_V1.63 --job_name=pi0_fighter_V1.63 --policy.device=cuda --wandb.enable=true --policy.push_to_hub=false --steps=30000 --policy.pretrained_path=lerobot/pi0_base --policy.compile_model=true --policy.dtype=bfloat16 --batch_size=8 --policy.gradient_checkpointing=true --policy.freeze_vision_encoder=true --policy.train_expert_only=true

echo "========== Starting pi0_fast =========="
lerobot-train --dataset.repo_id=training/fighter_V1.63 --policy.type=pi0_fast --output_dir=outputs/train/pi0_fast_fighter_V1.63 --job_name=pi0_fast_fighter_V1.63 --policy.device=cuda --wandb.enable=true --policy.push_to_hub=false --steps=30000 --policy.pretrained_path=lerobot/pi0fast_base --policy.compile_model=true --policy.dtype=bfloat16 --batch_size=8 --policy.gradient_checkpointing=true 

echo "========== Starting pi05 =========="
lerobot-train --dataset.repo_id=training/fighter_V1.63 --policy.type=pi05 --output_dir=outputs/train/pi05_fighter_V1.63 --job_name=pi05_fighter_V1.63 --policy.device=cuda --wandb.enable=true --policy.push_to_hub=false --steps=30000 --policy.pretrained_path=lerobot/pi05_base --policy.compile_model=true --policy.dtype=bfloat16 --batch_size=8 --policy.gradient_checkpointing=true --policy.freeze_vision_encoder=true --policy.train_expert_only=true

echo "========== All 3 done =========="
