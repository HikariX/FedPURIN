import os
import torch

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling, data_vis
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler, get_launcher
import time, copy
import numpy as np
from torch.utils.data import DataLoader

def personalized_model_pipeline_fedlips(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for personalized federated learning. Under this setting, models of each client is different.
       The final performance of is calculated by averaging the local model in each client.
       Typically, each local model is tested using local test dataset.
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed=seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset.
    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)
    # Split dataset into clients.
    train_sets = data_sampling(train_set, args, seed, train=True)
    test_sets = data_sampling(test_set, args, seed, train=False)

    ## visual data
    # data_vis(train_sets, args)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)
    for i in range(args.client.client_num):
        group.append(get_client(args=args, client_id=i, train_dataset=train_sets[i], test_dataset=test_sets[i]))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.optimizer.lr, args=args.learn.scheduler)

    # Setup launcher.
    launcher = get_launcher(args)

    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        start_time = time.time()
        # Initialize variable monitor.
        train_monitor = VariableMonitor()

        # Random sample participated clients in each communication round.
        # participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)
        if i == 0:
            participated_clients = list(range(args.client.client_num))  # 第一轮全选
        else:
            participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)  # 第二轮开始采样

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)

        args.learn.local_eps = args.learn.local_eps_first
        if i % args.dst.update_freq == 0:
            update_d = args.dst.update_d + (1.0 - args.dst.update_d) * (i / int(args.learn.global_eps))
            args.learn.local_eps_first_d = min(update_d, 1.0)
        else:
            args.learn.local_eps_first_d = 1.0

        # Local training for each participated client and add results to the monitor.
        # Use multiprocessing for acceleration.
        train_results = launcher.launch(
            clients=[group.clients[j] for j in participated_clients], lr=cur_lr, local_critic=args.learn.tau, task_name='train'
        )
        for item in train_results:
            train_monitor.append(item)


#         ## training loss for each epoch
#         sum_train_loss_epochs = [0] * len(train_results[0]['train_loss_epochs'])
#         for item in train_results:
#             for inn, loss in enumerate(item['train_loss_epochs']):
#                 sum_train_loss_epochs[inn] += loss

#         # Divide each element by the number of dictionaries to get the average
#         avg_train_loss_epochs = [total / len(train_results) / 100 for total in sum_train_loss_epochs]

#         for innel_epoch, loss_value in enumerate(avg_train_loss_epochs):
#             training_epoch = i*len(avg_train_loss_epochs) + innel_epoch
#             logger.logging(f"Training epoch {training_epoch}: loss_value:{loss_value:.3f}")


        # Testing
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()

            # Testing for each client and add results to the monitor
            # Use multiprocessing for acceleration.
            test_results = launcher.launch(
                clients=[group.clients[j] for j in range(args.client.client_num)], task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)

            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()

            # Logging test variables.
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=mean_test_variables, rnd=i)

#         for p in participated_clients: # Maybe the util function is also modified in LIPS. Thus, I commented it here.
#             results_train = train_monitor.get_results()
#             results_before = test_monitor.get_results()

#             acc_train = results_train[0][p]
#             loss_train = results_train[1][p]

#             acc_before = results_before[0][p]
#             loss_before = results_before[1][p]

#             logger.logging(f"Overall results for client {p}; acc_train: {acc_train:.3f}; loss_train:{loss_train:.3f}; acc_before: {acc_before:.3f}; loss_before:{loss_before:.3f}")


        # Aggregate parameters in each client.
        if args.dst.name == 'local':
            trans_cost = 0
        else:
            trans_cost = group.aggregate(i)

        # Logging for train variables.
        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update({'trans_cost(MB)': trans_cost / 1e6, 'lr': cur_lr})
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)

        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_monitor = VariableMonitor()

            # Testing for each client and add results to the monitor
            # Use multiprocessing for acceleration.
            test_results = launcher.launch(
                clients=[group.clients[j] for j in range(args.client.client_num)], task_name='test'
            )
            for item in test_results:
                test_monitor.append(item)

            # Get mean results across each client.
            mean_test_variables = test_monitor.variable_mean()

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=mean_test_variables, rnd=i)

            # Saving model checkpoints.
            # torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))
            results_after = test_monitor.get_results()


        ### visualize sparsity; local mask similarity
        all_client_grads = {}
        if i % args.dst.sparsity_visual == 0:
            for p in participated_clients:
                total_nonzero = 0.0
                total_params = 0.0
                client_grads = {}
                for name, param in group.clients[p].model.named_parameters():
                    if name not in group.clients[p].local_masks:
                        continue
                    total_nonzero += (torch.abs(param) > 1e-8).float().sum().item()
                    total_params += param.numel()

                    ### grads
                    grad_list = [grad[name] for grad in group.clients[p].local_grad if name in grad]

                    grad_norm = torch.norm(torch.stack([g.view(-1) for g in grad_list]), p=2).item()
                    mean_grad = torch.mean(torch.stack(grad_list), dim=0)
                    grad_diversity = torch.mean(torch.stack([torch.norm(g - mean_grad, p=2) for g in grad_list])).item()

                    client_grads[name] = [grad_norm, grad_diversity]

                    if name not in all_client_grads:
                        all_client_grads[name] = []

                    all_client_grads[name].append((grad_norm, grad_diversity))


                acc_after = results_after[0][p]
                loss_after = results_after[1][p]
                density = total_nonzero / total_params if total_params > 0 else 0

                logger.logging(f"Overall sparsity for client {p}: {density}, acc_after: {acc_after:.3f}; loss_after:{loss_after:.3f}")

            for name in all_client_grads:
                if all_client_grads[name]:  # Ensure there's data for the layer
                    mean_grad_norm = np.mean([norm for norm, _ in all_client_grads[name]])
                    mean_grad_div = np.mean([div for _, div in all_client_grads[name]])

                    # Log each layer's mean separately
                    layer_log = f"{name}: mean grad norm: {mean_grad_norm:.3f}, mean grad div: {mean_grad_div:.3f}"
                    logger.logging(layer_log)



    # Fine-tuning
    # Fine-tune model on each client and collect all the results.
    finetune_results = launcher.launch(
        clients=[group.clients[j] for j in range(args.client.client_num)],
        lr=cur_lr,
        finetune_args=args.learn.finetune_parameters,
        task_name='finetune'
    )

    # Logging fine-tune results
    logger.logging('=' * 60)
    logger.logging("#### mean final results after finetune ####")
    for key in finetune_results[0][0].keys():
        for eid in range(len(finetune_results[0])):
            tmp_mean = sum([finetune_results[cid][eid][key]
                            for cid in range(len(finetune_results))]) / len(finetune_results)
            # logger.add_scalar(f'finetune/{key}', tmp_mean, eid)
            logger.logging(f'finetune/{key}: {tmp_mean}')