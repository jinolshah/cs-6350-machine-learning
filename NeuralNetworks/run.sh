#!/bin/bash
echo -e "1 - Manual Neural Network \n2 - SGD \n3 - Pytorch"
read -p 'Choice: ' choice

if [ "$choice" == 1 ]; then
    python fb_pass.py
fi

if [ "$choice" == 2 ]; then
    python sgd_neural_net.py
fi

if [ "$choice" == 3 ]; then
    python torch_neural_network.py
fi