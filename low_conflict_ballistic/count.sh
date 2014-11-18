#!/bin/bash
for file in $( find . -name 'game_*'); do
    echo $file\n
    grep ',90,' $file | wc -l
done