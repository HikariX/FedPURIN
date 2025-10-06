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
    
    parser.add_argument('--tau', type=float, default=0.5, help='The ratio of local critical parameters (default=0.0, without using local weights)')
    parser.add_argument('--beta', type=int, default=100, help='default: 100')



    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--isLocal', type=str2bool, default=True, help='Is the non-critical param defined by local model.')
    parser.add_argument('--isGrad', type=str2bool, default=True, help='Is the non-critical param defined by actual gradient.')
    parser.add_argument('--HeadEnd', type=str2bool, default=False, help='Is the first and last module get updated specially. True: all the params are critical. False: ordinary as others.')
    parser.add_argument('--isHessian', type=str2bool, default=True, help='Is the non-critical param defined by hessian.')
    parser.add_argument('--isBN', type=str2bool, default=False, help='Is the BN layer be processed.')
    parser.add_argument('--GPU', type=str, default='0', help='GPU')
    parser.add_argument('--exc', type=str, default='all', help='How to exclude special parameters. HeadEnd: do not process the first and last layer; BN: exclude bn as FedBN does.')

    return parser.parse_args()

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

if args.exc == 'BN':
    aggr_param = dict(name='except', keywords=['bn', 'downsample.1'])
elif args.exc == 'HeadEnd':
    aggr_param = dict(name='except', keywords=['pre', 'fc'])
elif args.exc == 'Both':
    aggr_param = dict(name='except', keywords=['pre', 'bn', 'downsample.1', 'fc'])
elif args.exc == 'all':
    aggr_param = dict(name='all', )             


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
        finetune_parameters=dict(name='all'),
        # test_place=['after_aggregation', 'before_aggregation'],
        test_place=['before_aggregation'],
        tau=args.tau,  # the ratio of critical parameters (tau) in FedCAC
        beta=args.beta,  # used to control the collaboration of critical parameters
    ),
    model=dict(
        name=args.model,
        input_channel=1,
        class_number=10,
    ),
    client=dict(name='purin_client', client_num=args.client_num),
    server=dict(name='base_server'),
    group=dict(
        name='purin_group',
        aggregation_method='avg',
        aggregation_parameters=aggr_param,
        isLocal=args.isLocal,
        include_non_param=False
    ),
    hyper=dict(
        HeadEnd=args.HeadEnd,
        isHessian=args.isHessian,
        isGrad=args.isGrad,
        isBN=args.isBN,
        seed=args.seed,
        exc=args.exc
    ),
    other=dict(test_freq=1, logging_path=f'./logging/fmnist_purin_resnet/{args.noniid}_{args.alpha}_{args.tau}_{args.exc}_{args.seed}_{args.HeadEnd}_{args.isLocal}_{args.isHessian}_{args.isGrad}_{args.client_num}')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline
    
    print(exp_args)
    personalized_model_pipeline(exp_args, seed=args.seed)
