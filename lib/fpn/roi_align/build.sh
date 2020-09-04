#!/usr/bin/env bash
cuda_path=/usr/local/cuda-9.0/

cd src/cuda
echo "Compiling stnn kernels by nvcc..."
nvcc -c -o  roi_align.cu.o  roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py

