for alpha in 0.1 0.5 1.0
do
    for seed in 1 2 3
    do
        nohup python -u flzoo/cifar10/cifar10_purin_resnet_config.py --GPU 0 --seed ${seed} --alpha ${alpha} --isBN True --exc BN >./purin_a${alpha}_${seed}.log 2>&1 &
    done
done