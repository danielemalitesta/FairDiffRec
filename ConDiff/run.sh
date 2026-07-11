# #!/bin/bash

dataset_name=$1

case $dataset_name in
    "ml-1m")
        cmd="python main.py --cuda --dataset=ml-1m --data_path=../datasets/ --lr=0.0005 --weight_decay=0.01 --dims=[200,200] --steps=20 --noise_scale=0.001 --noise_min=0.02 --noise_max=0.2 --lamd1 0.1 --lamd2 0.1"
        ;;
    "foursquare_tky")
        cmd="python main.py --cuda --dataset=foursquare_tky --data_path=../datasets/ --lr=5e-05 --weight_decay=0.01 --dims=[1000] --steps=20 --noise_scale=0.001 --noise_min=0.001 --noise_max=0.01 --lamd1 0.1 --lamd2 0.1"
        ;;
    "books")
        cmd="python main.py --cuda --dataset=books --data_path=../datasets/ --lr=5e-05 --weight_decay=0.01 --dims=[1000] --steps=20 --noise_scale=0.001 --noise_min=0.001 --noise_max=0.2 --lamd1 0.1 --lamd2 0.1"
        ;;
    "lastfm")
        cmd="python main.py --cuda --dataset=lastfm --data_path=../datasets/ --lr=0.0005 --weight_decay=0.01 --dims=[200,200] --steps=10 --noise_scale=0.001 --noise_min=0.001 --noise_max=0.01 --lamd1 0.1 --lamd2 0.1"
        ;;
    *)
        echo "Invalid dataset name! Please provide one of the following dataset names: ml-1m, foursquare_tky, books, lastfm"
        exit 1
        ;;
esac

echo $cmd
eval "$cmd"
