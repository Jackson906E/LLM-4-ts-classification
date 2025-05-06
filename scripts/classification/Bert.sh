export CUDA_VISIBLE_DEVICES=0

TASK_NAME=classification_llm
OUT_NAME=BertEmbed*TimesNet+loss

# python -u run.py \
#   --task_name $TASK_NAME \
#   --is_training 1 \
#   --root_path ./dataset/EthanolConcentration/ \
#   --model_id EthanolConcentration \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10 > log/LLM/$OUT_NAME-EthanolConcentration.log 2>&1

# python -u run.py \
#   --task_name $TASK_NAME \
#   --is_training 1 \
#   --root_path ./dataset/FaceDetection/ \
#   --model_id FaceDetection \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 64 \
#   --d_ff 256 \
#   --top_k 3 \
#   --num_kernels 4 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10 > log/LLM/$OUT_NAME-FaceDetection.log 2>&1

python run.py \
--task_name $TASK_NAME \
--is_training 1 \
--root_path ./dataset/Handwriting/ \
--model_id Handwriting \
--model TimesNet \
--data UEA \
--e_layers 2 \
--batch_size 16 \
--d_model 32 \
--d_ff 64 \
--top_k 3 \
--des 'Exp' \
--itr 1 \
--learning_rate 0.001 \
--train_epochs 30 \
--patience 10 > log/LLM/$OUT_NAME-Handwriting.log 2>&1

# python -u run.py \
#   --task_name $TASK_NAME \
#   --is_training 1 \
#   --root_path ./dataset/Heartbeat/ \
#   --model_id Heartbeat \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --top_k 1 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10 > log/Bert-Heartbeat.log 2>&1

python -u run.py \
  --task_name $TASK_NAME \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
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
  --train_epochs 60 \
  --patience 10 > log/LLM/$OUT_NAME-JapaneseVowels.log 2>&1

# python -u run.py \
#   --task_name $TASK_NAME \
#   --is_training 1 \
#   --root_path ./dataset/PEMS-SF/ \
#   --model_id PEMS-SF \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 6 \
#   --batch_size 16 \
#   --d_model 64 \
#   --d_ff 64 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10 > log/Bert-PEMS-SF.log 2>&1

# python -u run.py \
#   --task_name $TASK_NAME \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP1/ \
#   --model_id SelfRegulationSCP1 \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10 > log/Bert-SelfRegulationSCP1.log 2>&1

python -u run.py \
  --task_name $TASK_NAME \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 > log/LLM/$OUT_NAME-SelfRegulationSCP2.log 2>&1

# python -u run.py \
#   --task_name $TASK_NAME \
#   --is_training 1 \
#   --root_path ./dataset/SpokenArabicDigits/ \
#   --model_id SpokenArabicDigits \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 2 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10 > log/Bert-SpokenArabicDigits.log 2>&1

python -u run.py \
  --task_name $TASK_NAME \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 > log/LLM/$OUT_NAME-UWaveGestureLibrary.log 2>&1
