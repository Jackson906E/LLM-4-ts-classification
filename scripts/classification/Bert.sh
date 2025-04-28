export CUDA_VISIBLE_DEVICES=0

TASK_NAME=classification_llm

python -u run.py \
  --task_name $TASK_NAME \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 

python -u run.py \
  --task_name $TASK_NAME \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10