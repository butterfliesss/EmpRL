python run_dialogue.py \
    --model_name_or_path /home/huima/data_ssd/huggingface_data/t5-base \
    --do_train \
    --do_eval \
    --train_file data/empatheticdialogues/sft/train.json \
    --validation_file data/empatheticdialogues/sft/valid.json \
    --test_file data/empatheticdialogues/sft/test.json \
    --context_column input \
    --response_column output \
    --output_dir outputs/empathetic/finetune-t5base \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --overwrite_output_dir \
    --max_source_length 200 \
    --max_target_length 50 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="linear" \
    --num_train_epochs=3 \
    --save_strategy="epoch" \
    --evaluation_strategy="epoch" \
    --save_total_limit=1 \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_loss" \
    --generation_max_length 30 \
    --seed 42 \
 >> empathetic_finetune_t5base.log 2>&1 &
