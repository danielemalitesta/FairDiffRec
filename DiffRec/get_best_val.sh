#!/bin/bash

grep ValRecall logs/$1/*.log | awk '{print $1, $5}' | sed 's/.log:\[Valid\]://' | awk '{split($2,a,"-"); print $1, a[2]}' | sort -k2n