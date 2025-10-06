for alpha in 1.0
do
    for seed in 1 2 3
    do
        nohup python -u flzoo/cifar10/cifar10_fedbn_resnet_config.py --GPU 1 --alpha ${alpha} --seed ${seed} >fedbn_a${alpha}_${seed}.log 2>&1 &
    done
done