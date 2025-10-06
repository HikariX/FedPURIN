for alpha in 0.1 0.5
do
    for seed in 1 2 3
    do
        nohup python -u flzoo/cifar10/cifar10_pfedsd_resnet_config.py --GPU 2 --alpha ${alpha} --seed ${seed} >./results_c10/pfedsd_a${alpha}_${seed}.log 2>&1 &
    done
done