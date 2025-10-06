for alpha in 1.0
do
    for seed in 1 2 3
    do
        nohup python -u flzoo/cifar10/cifar10_fedavg_resnet_config.py --GPU 3 --alpha ${alpha} --seed ${seed} >fedavg_a${alpha}_${seed}.log 2>&1 &
    done
done