#!/bin/bash
for file in $( ls ./game_*); do #find . -name 'game_*'); do
    echo $file\n
    grep ',270,' $file | wc -l
done
