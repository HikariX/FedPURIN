import argparse
from easydict import EasyDict
import sys
from pathlib import Path
import os

def str2bool(str):
    return True if str.lower() == 'true' else False

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Arguments")
    
    ## model
    parser.add_argument('--model', type=str, default='resnet8', help='model name')

    ## data
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Dirichlet distribution alpha parameter for non-IID data')
    parser.add_argument('--noniid', type=str, default='dirichlet', help='Non-IID data sampling method')
    parser.add_argument('--train_num', type=int, default=500, help='The number of samples in each client')
    parser.add_argument('--client_num', type=int, default=20, help='The number of clients')

    parser.add_argument('--seed', type=int, default=2, help='Random seed')

    parser.add_argument('--GPU', type=str, default='0', help='GPU')

    return parser.parse_args()

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

exp_args = dict(
    data=dict(
        dataset='fmnist',
        data_path='./data/fmnist',
        sample_method=dict(name=args.noniid, alpha=args.alpha, train_num=args.train_num, test_num=100)
    ),
    learn=dict(
        device='cuda:0', 
        local_eps=5, 
        global_eps=200, 
        batch_size=100, 
        optimizer=dict(name='sgd', lr=0.1, momentum=0.9),
        # Only fine-tune parameters whose name contain the keyword "fc".
        finetune_parameters=dict(name='contain', keywords=['fc']),
        test_place=['after_aggregation', 'before_aggregation']
    ),
    model=dict(
        name='resnet8',
        input_channel=1,
        class_number=10,
    ),
    client=dict(name='base_client', client_num=args.client_num),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        # Only aggregate parameters whose name does not contain the keyword "fc".
        aggregation_parameters=dict(
            name='except',
            # The name of parameter that contain these two keys is a BN layer.
            # For other types of networks, please manually identify the key of BN layers.
            keywords=['bn', 'downsample.1'],
        ),
        include_non_param=False
    ),
    other=dict(test_freq=1, logging_path=f'./logging/fmnist_fedbn_resnet/{args.noniid}_{args.client_num}_{args.alpha}_{args.seed}')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline
    print(exp_args)
    personalized_model_pipeline(exp_args, seed=args.seed)
