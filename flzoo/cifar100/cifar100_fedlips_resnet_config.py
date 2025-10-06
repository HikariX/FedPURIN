import argparse
from easydict import EasyDict

def str2bool(str):
    return True if str.lower() == 'true' else False

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Arguments")
    parser.add_argument('--beta', type=int, default=170,
                        help='Used to control the collaboration of critical parameters')
    ## model
    parser.add_argument('--model', type=str, default='ResNet8', help='model name') ## vgg11_bn, ResNet8


    ## data
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Dirichlet distribution alpha parameter for non-IID data')
    parser.add_argument('--noniid', type=str, default='dirichlet', help='Non-IID data sampling method')
    parser.add_argument('--train_num', type=int, default=100, help='The number of samples in each client')
    parser.add_argument('--client_num', type=int, default=40, help='The number of samples in each client')
    parser.add_argument('--local_eps_first', type=int, default=5, help='The number of samples in each client')
    parser.add_argument('--local_eps_first_d', type=int, default=1.0, help='The number of samples in each client')

    ## dst
    parser.add_argument('--dst', type=str, default='rigl', help='default: rigl (with sparsity)')  ## rigl, local
    parser.add_argument('--tau', type=float, default=0.0,
                        help='The ratio of local critical parameters (default=0.0, without using local weights)')
    parser.add_argument('--density_init', type=float, default=1.0, help='default: 1.0 (dense initialization)')
    parser.add_argument('--prune_method', type=str, default='magnitude',
                        help='pruning method on client')  ## sensity, magnitude, random
    parser.add_argument('--regrow_method', type=str, default='topk',
                        help='regrow method on server')  ## topk, lowk, randomk

    parser.add_argument('--prune_ll', type=str, default='large', help='pruning from large to small')
    parser.add_argument('--update_mode', type=str, default='global_sel',
                        help='how to make use of server model')  ## global_sel, global_sel_init, local, fdst

    parser.add_argument('--no_margin', type=str2bool, default=True,
                        help='defaul: True (without include first and last layer)')
    parser.add_argument('--update_d', type=float, default=0.1, help='init density ratio (will increase to 1.0)')
    parser.add_argument('--update_freq', type=int, default=10, help='The frequency for masking')

    parser.add_argument('--seed', type=int, default=2, help='Random seed')

    return parser.parse_args()


args = parse_args()

exp_args = dict(
    data=dict(
        dataset='cifar100',
        data_path='./data/CIFAR100',
        sample_method=dict(name=args.noniid, alpha=args.alpha, train_num=args.train_num, test_num=400),
    ),
    learn=dict(
        device='cuda:0',
        local_eps=5,
        global_eps=300,
        batch_size=100,
        optimizer=dict(name='sgd', lr=0.1, momentum=0.9),  #, weight_decay=1e-3
        finetune_parameters=dict(name='all'),
        test_place=['after_aggregation', 'before_aggregation'],
        tau=args.tau,  # the ratio of critical parameters in FedCAC
        beta=args.beta,  # used to control the collaboration of critical parameters
        local_eps_first=args.local_eps_first,
        local_eps_first_d=args.local_eps_first_d,

    ),
    model=dict(
        name=args.model,
        input_channel=3,
        class_number=100,
    ),
    client=dict(name='fedcac_client', client_num=args.client_num),
    server=dict(name='base_server'),
    group=dict(
        name='fedcac_group',
        aggregation_method='avg',
        aggregation_parameters=dict(name='all', ),
    ),

    dst=dict(
        name=args.dst,  ## local,
        density_init=args.density_init,
        sparsity_visual=1,
        prune_method=args.prune_method,
        prune_ll=args.prune_ll,
        regrow_method=args.regrow_method,
        death_rate=args.death_rate,
        no_margin=args.no_margin,
        update_d=args.update_d,
        update_freq=args.update_freq,
    ),

    other=dict(test_freq=1, logging_path=f'./logging/cifar100_fedcac_resnet/{args.noniid}_{args.alpha}_{args.tau}_{args.beta}_{args.dst}_{args.density_init}_{args.seed}')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import personalized_model_pipeline
    print(exp_args)

    personalized_model_pipeline(exp_args, seed=args.seed)
