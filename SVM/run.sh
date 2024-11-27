#!/bin/bash
echo -e "1: Primal SVM \n2: Dual SVM \n3: Gaussian Kernel SVM \n4: Kernel Perceptron"
read -p 'Choice: ' choice

if [ "$choice" == 1 ]; then
    python primal.py
fi

if [ "$choice" == 2 ]; then
    python dual.py
fi

if [ "$choice" == 3 ]; then
    python gaussian.py
fi

if [ "$choice" == 4 ]; then
    python kernel_perceptron.py
fi