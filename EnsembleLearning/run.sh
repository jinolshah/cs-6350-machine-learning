#!/bin/bash
echo "1 - adaboost\n2 - bagging\n3 - 100 bagged predictors\n4 - Ranfom Forest\n5 - 100 Random Forests"
read -p 'Algo: ' algo

if [ "$algo" == 1 ]; then
    python adaboost.py
fi

if [ "$algo" == 2 ]; then
    python bagging.py
fi

if [ "$algo" == 3 ]; then
    python bagging_100.py
fi

if [ "$algo" == 4 ]; then
    python randomForest.py
fi

if [ "$algo" == 5 ]; then
    python randomForest_100.py
fi
