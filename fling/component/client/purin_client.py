import copy
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from fling.utils import get_optimizer, VariableMonitor, get_weights
from .base_client import BaseClient


@CLIENT_REGISTRY.register('purin_client')
class PurinClient(BaseClient):
    """
    Overview:
        This class is the base implementation of client in Purin.
    """

    def __init__(self, args, client_id, train_dataset, test_dataset=None):
        """
        Initializing train dataset, test dataset(for personalized settings).
        """
        super(PurinClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.critical_parameter = None  # record the critical parameter positions in Purin
        self.customized_model = copy.deepcopy(self.model)  # customized global model
        self.local_grad = {}
        self.round_recorder = 0
        self.prev_local_model = None
    
    def train(self, lr, device=None, train_args=None):
        """
        Local training.
        """
        # record the model before local updating, used for critical parameter selection
        initial_model = copy.deepcopy(self.model)
        
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.train()
        self.model.to(self.device)

        # Set optimizer, loss function.
        if train_args is None:
            weights = self.model.parameters()
        else:
            weights = get_weights(self.model, parameter_args=train_args)
        op = get_optimizer(weights=weights, **self.args.learn.optimizer)

        # Set the loss function.
        criterion = nn.CrossEntropyLoss()

        monitor = VariableMonitor()
        # Main training loop.
        for epoch in range(self.args.learn.local_eps):
            for _, data in enumerate(self.train_dataloader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                self.train_step(batch_data=preprocessed_data, criterion=criterion, monitor=monitor, optimizer=op)
            
            # get local gradient of the last round
            if epoch == self.args.learn.local_eps - 1:
                if self.args.hyper.isGrad:
                    self.local_grad = {}
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            self.local_grad[name] = param.grad.clone().detach().to('cpu').half()
                
        # Calculate the mean metrics.
        mean_monitor_variables = monitor.variable_mean()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        if device is not None:
            self.device = device_bak

        # select the critical parameters
        if self.args.hyper.isGrad:
            self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter_grad(
                model=self.model, local_grad=self.local_grad, tau=self.args.learn.tau
            )
        else:
            self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
                prevModel=initial_model, model=self.model, tau=self.args.learn.tau
            )
        self.prev_local_model = copy.deepcopy(self.model) # Save the whole model before sparsification.
        self.prev_local_model.to('cpu')
        self.sparsify_model(self.model, self.local_mask, self.args.learn.tau)
        self.round_recorder += 1
        return mean_monitor_variables

    def evaluate_critical_parameter(self, prevModel: nn.Module, model: nn.Module,
                                    tau: int) -> Tuple[torch.Tensor, list, list]:
        r"""
        Overview:
            Implement critical parameter selection.
        """
        global_mask = []  # mark non-critical parameter
        local_mask = []  # mark critical parameter
        critical_parameter = []

        self.model.to(self.device)
        prevModel.to(self.device)

        # select critical parameters in each layer
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            if self.args.hyper.exc in ['BN', 'Both']:
                if 'bn' in name2 or 'downsample.1' in name2:
                    continue # Jump over all the bn layers.
            
            if 'pre' in name2 or 'fc' in name2:  # Head or nn.Linear
                if self.args.hyper.HeadEnd:
                    mask = torch.ones_like(param.data).to('cpu') # 不处理头尾，故全传
                    global_mask.append(torch.zeros_like(mask).to('cpu'))
                    local_mask.append(mask)
                    critical_parameter.append(mask.view(-1))
                    continue
            elif 'bn' in name2 or 'downsample.1' in name2:
                if self.args.hyper.isBN: # bn参数全传，若为false则进行挑选合作
                    mask = torch.ones_like(param.data).to('cpu')
                    global_mask.append(torch.zeros_like(mask).to('cpu'))
                    local_mask.append(mask)
                    critical_parameter.append(mask.view(-1))
                    continue

            g = (param.data - prevparam.data)
            v = param.data
            
            if self.args.hyper.isHessian:
                c = torch.abs(-g * v + 0.5 * torch.pow(g * v, 2))
            else:
                c = torch.abs(g * v)
                
            metric = c.view(-1)
            num_params = metric.size(0)
            nz = int(tau * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            # if threshold equals 0, select minimal nonzero element as threshold
            # notice that in fedcac, the thresh and new value is not equal, a mismatch! We fixed it and regards all values less than or equal 1e-10 as 0.
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-10]
                if len(new_metric) == 0:  # this means all items in metric are zero
                    print(f'Abnormal!!! metric:{metric}')
                else:
                    thresh = new_metric.sort()[0][0]

            # Get the local mask and global mask
            mask = (c >= thresh).int().to('cpu')
            global_mask.append((c < thresh).int().to('cpu'))
            local_mask.append(mask)
            critical_parameter.append(mask.view(-1))
        model.zero_grad()
        critical_parameter = torch.cat(critical_parameter)

        self.model.to('cpu')
        prevModel.to('cpu')

        return critical_parameter, global_mask, local_mask
    
    def evaluate_critical_parameter_grad(self, model: nn.Module, local_grad: dict,
                                    tau: int) -> Tuple[torch.Tensor, list, list]:
        r"""
        Overview:
            Implement critical parameter selection.
        """
        global_mask = []  # mark non-critical parameter
        local_mask = []  # mark critical parameter
        critical_parameter = []

        self.model.to(self.device)

        # select critical parameters in each layer
        for name, param in model.named_parameters():
            if self.args.hyper.exc in ['BN', 'Both']:
                if 'bn' in name or 'downsample.1' in name:
                    continue # Jump over all the bn layers.
                    
            if 'pre' in name or 'fc' in name:  # Head or nn.Linear
                if self.args.hyper.HeadEnd:
                    mask = torch.ones_like(param.data).to('cpu') # 不处理头尾，故全传
                    global_mask.append(torch.zeros_like(mask).to('cpu'))
                    local_mask.append(mask)
                    critical_parameter.append(mask.view(-1))
                    continue
            elif 'bn' in name or 'downsample.1' in name:
                if self.args.hyper.isBN:
                    mask = torch.ones_like(param.data).to('cpu')
                    global_mask.append(torch.zeros_like(mask).to('cpu'))
                    local_mask.append(mask)
                    critical_parameter.append(mask.view(-1))
                    continue
                    
            g = local_grad[name].to(self.device)
            v = param.data
            
            if self.args.hyper.isHessian:
                c = torch.abs(-g * v + 0.5 * torch.pow(g * v, 2))
            else:
                c = torch.abs(g * v)

            metric = c.view(-1)
            num_params = metric.size(0)
            nz = int(tau * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            # if threshold too small, select a relatively reasonable element as threshold
            # notice that in fedcac, the thresh is different with new value. We fixed it and regards all values less than or equal 1e-10 as 0.
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-10]
                if len(new_metric) == 0:  # this means all items in metric are zero
                    print(f'Abnormal!!! param_name:{name},metric:{metric}')
                else:
                    # print('$' * 100, thresh, copy.deepcopy(new_metric).sort()[0][0])
                    thresh = new_metric.sort()[0][0]
                    # print('$' * 100, thresh, new_metric.sort()[0][0])

            # Get the local mask and global mask
            mask = (c >= thresh).int().to('cpu')
            global_mask.append((c < thresh).int().to('cpu'))
            local_mask.append(mask)
            critical_parameter.append(mask.view(-1))
        model.zero_grad()
        critical_parameter = torch.cat(critical_parameter)

        self.model.to('cpu')

        return critical_parameter, global_mask, local_mask
    
    def sparsify_model(self, model: nn.Module, local_mask: list, tau: int):
        self.model.to(self.device)
        index = 0
        for name, param in model.named_parameters():
            if self.args.hyper.exc in ['BN', 'Both']:
                if 'bn' in name or 'downsample.1' in name:
                    continue # Jump over all the bn layers.
                    
            mask = self.local_mask[index].to('cpu')
            param.data = param.data * mask.to(self.args.learn.device)
            index += 1
        
        self.model.zero_grad()
        self.model.to('cpu')
            
