#!/usr/bin/env bash
cuda_path=/usr/local/cuda-9.0/

cd src
echo "Compiling stnn kernels by nvcc..."
nvcc -c -o highway_lstm_kernel.cu.o highway_lstm_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ..
python build.py

