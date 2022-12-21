# baseline cross-entropy -> 0.9061
#python train.py --is_train --use_amp --device 1 --wandb --method ce --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/ce'
# baseline cross-entropy + supervised contrastive loss -> 0.9061
#python train.py --is_train --use_amp --device 1 --wandb --method scl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/scl'
# baseline cross-entropy + dual contrastive loss -> 0.4973
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/dualcl'
# cross-entropy + dual contrastive loss + use_amp False -> 0.5391
#python train.py --is_train --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 16 --output_path './model/saved_model/dualcl_without_use_amp'

# cross-entropy + use token_type_ids -> 0.9061
#python train.py --is_train --use_amp --device 1 --wandb --method ce --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/ce_with_token_type_ids'
# cross-entropy + supervised contrastive loss + use token_type_ids -> 0.9061
#python train.py --is_train --use_amp --device 1 --wandb --method scl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/scl_with_token_type_ids'
# cross-entropy + dual contrastive loss + use token_type_ids -> 0.4973
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/dualcl_with_token_type_ids'

# cross-entropy + dual contrastive loss + use basic position_ids -> 0.9048
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/dualcl_with_token_type_ids_with_basic_position_ids'

# cross-entropy + dual contrastive loss + use basic position_ids + without special token -> 0.9043
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --output_path './model/saved_model/dualcl_with_token_type_ids_with_basic_position_ids_without_special_token'

# cross-entropy + dual contrastive loss + use basic position_ids + alpha 값 조절 (0.01) -> 0.9046
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --alpha 0.01 --output_path './model/saved_model/dualcl_with_token_type_ids_with_basic_position_ids_alpha0.01'
# cross-entropy + dual contrastive loss + use basic position_ids + alpha 값 조절 (0.05) -> 0.9051
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --alpha 0.05 --output_path './model/saved_model/dualcl_with_token_type_ids_with_basic_position_ids_alpha0.05'
# cross-entropy + dual contrastive loss + use basic position_ids + alpha 값 조절 (0.1) -> 0.9047
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --alpha 0.1 --output_path './model/saved_model/dualcl_with_token_type_ids_with_basic_position_ids_alpha0.1'

# cross-entropy + dual contrastive loss + use basic position_ids + alpha 값 조절 (0.05) + lr 2e-5 -> 0.9076
#python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --alpha 0.05 --lr 2e-5 --output_path './model/saved_model/dualcl_with_token_type_ids_with_basic_position_ids_alpha0.05_lr2e-5'

# cross-entropy + dual contrastive loss + use basic position_ids + alpha 값 조절 (0.05) + lr 2e-5 + early stop patience 50->
python train.py --is_train --use_amp --device 1 --wandb --method dualcl --model_name tbert --epochs 10 --eval_steps 1000 --train_batch_size 32 --alpha 0.05 --lr 2e-5 --patience 50 --output_path './model/saved_model/dualcl_with_token_type_ids_with_basic_position_ids_alpha0.05_lr2e-5_patience50'


