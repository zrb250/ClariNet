#!/usr/bin/env bash

#echo "Preprocessing (Preparing Mel Spectrogram)"
#python3 preprocessing.py --in_dir ljspeech --out_dir DATASETS/ljspeech

echo "Step 3. Train Gaussian Autoregressive WaveNet (Teacher)"
python3 train.py --model_name wavenet_gaussian --batch_size 8 --num_blocks 2 --num_layers 10

echo "Step 4. Synthesize (Teacher)"
python3 synthesize.py --model_name wavenet_gaussian --num_blocks 2 --num_layers 10 --load_step 10318 --num_samples 5
python3 synthesize.py --model_name wavenet_gaussian --num_blocks 2 --num_layers 10 --load_step 92862 --num_samples 5
python3 synthesize.py --model_name wavenet_gaussian --num_blocks 2 --num_layers 10 --load_step 203412 --num_samples 5
python3 synthesize.py --model_name wavenet_gaussian --num_blocks 2 --num_layers 10 --load_step 302170 --num_samples 5
python3 synthesize.py --model_name wavenet_gaussian --num_blocks 2 --num_layers 10 --load_step 400928 --num_samples 5


echo "Step 5. Train Gaussian Inverse Autoregressive Flow (Student)"
python3 train_student.py --model_name wavenet_gaussian_student --teacher_name wavenet_gaussian --teacher_load_step 400928 --batch_size 2 --num_blocks_t 2 --num_layers_t 10 --num_layers_s 10 --KL_type qp

echo "Step 6. Synthesize (Student)"
python3 synthesize_student.py --model_name wavenet_gaussian_student --load_step 400928 --teacher_name wavenet_gaussian --teacher_load_step 10000 --num_blocks_t 2 --num_layers_t 10 --num_layers_s 10 --num_samples 5
