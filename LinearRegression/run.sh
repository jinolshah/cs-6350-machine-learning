#!/bin/bash
echo "1 - Batch GD 2 - Stochastic GD 3 - Optimal Weight vector calculation"
read -p 'Algo: ' algo

if [ "$algo" == 1 ]; then
    python batchGD.py
fi

if [ "$algo" == 2 ]; then
    python stochastic_GD.py
fi

if [ "$algo" == 3 ]; then
    python optimal_weight.py
fi
