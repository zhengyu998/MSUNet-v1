import torch
import numpy as np



class ParameterMasker(object):
    """
    Adapted from Neural network distiller https://github.com/NervanaSystems/distiller
    A ParameterMasker can mask a parameter tensor or a gradients tensor.

    It is used when pruning DNN weights.
    """
    def __init__(self, param_name):
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.is_regularization_mask = False
        self.use_double_copies = False
        self.mask_on_forward_only = False
        self.unmasked_copy = None
        self.backward_hook_handle = None

    def apply_mask(self, parameter):
        """Apply a mask on the weights tensor (parameter)."""
        if self.mask is None:
            return
        if self.use_double_copies:
            self.unmasked_copy = parameter.clone().detach()
        self.mask_tensor(parameter)
        if self.is_regularization_mask:
            self.mask = None
        return parameter

    def mask_tensor(self, tensor):
        if self.mask is not None:
            tensor.data.mul_(self.mask)

    def mask_gradient(self, gradient):
        if self.mask is not None:
            return gradient.mul(self.mask)

    def revert_weights(self, parameter):
        if not self.use_double_copies or self.unmasked_copy is None:
            return
        parameter.data.copy_(self.unmasked_copy)
        self.unmasked_copy = None



class Pruner(object):
    '''
    Adapted from Neural network distiller https://github.com/NervanaSystems/distiller
    :param object:
    :return:
    The mask should not be used on Ternary weights
    '''
    def __init__(self, model, start_epoch, end_epoch, sparsity):
        self.model = model
        self.mask_dict = self._get_mask_dict()
        self.start_epoch, self.end_epoch = start_epoch, end_epoch
        self.sparsity = sparsity

    def _get_mask_dict(self):
        weight_list = ['blocks.2.0.se.conv_reduce.weight',
                       'blocks.2.0.se.conv_expand.weight',
                       'blocks.2.1.se.conv_reduce.weight',
                       'blocks.2.1.se.conv_expand.weight',
                       'blocks.3.0.se.conv_reduce.weight',
                       'blocks.3.0.se.conv_expand.weight',
                       'blocks.3.1.se.conv_reduce.weight',
                       'blocks.3.1.se.conv_expand.weight']
        mask_dict = {}
        for k in weight_list:
            mask_dict[k] = ParameterMasker(k)
        return mask_dict


    def _get_current_sparsity(self, epoch):
        if epoch <= self.end_epoch:
            r = (epoch - self.start_epoch)/(self.end_epoch - self.start_epoch)
        else:
            r = 1
        s = self.sparsity - self.sparsity*(1-r)**3
        #print('Current Sparsity is {}'.format(s))
        return s

    def _update_all_mask(self, epoch):
        s = self._get_current_sparsity(epoch)
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                self._mask_to_target_sparsity(self.mask_dict[k], w.data, s)


    def _mask_to_target_sparsity(self, mask, weight, sparsity):
        bottomk, _ = torch.topk(weight.abs().view(-1), int(sparsity*weight.numel()), largest=False, sorted=True)
        if len(bottomk) > 0:
            threshold = bottomk.data[-1]
        else:
            threshold = 0
        if sparsity > 0:
            mask.mask = self._threshold_mask(weight, threshold).requires_grad_(False)
        else:
            mask.mask = torch.ones_like(weight).requires_grad_(False)
        # if mask.mask is None: # initialize mask
        #     mask.mask = torch.ones_like(weight).requires_grad_(False)


    def _threshold_mask(self, weight, threshold):
        return torch.gt(torch.abs(weight), threshold).type(weight.type())

    def on_epoch_begin(self,epoch):
        self._update_all_mask(epoch)

    def on_minibatch_begin(self):
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                self.mask_dict[k].apply_mask(w)

    def on_minibatch_end(self):
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                self.mask_dict[k].apply_mask(w)

    def print_statistics(self):
        for k, w in self.model.named_parameters():
            if k in self.mask_dict:
                #self.mask_dict[k].apply_mask(w)
                sparsity = (w.numel() - w.nonzero().size(0)) / w.numel()
                print('Pruner: {} with sparsity {}'.format(k, sparsity))

def print_sparsity_statistics(model, print=True):
    weight_list = ['blocks.2.0.se.conv_reduce.weight',
                   'blocks.2.0.se.conv_expand.weight',
                   'blocks.2.1.se.conv_reduce.weight',
                   'blocks.2.1.se.conv_expand.weight',
                   'blocks.3.0.se.conv_reduce.weight',
                   'blocks.3.0.se.conv_expand.weight',
                   'blocks.3.1.se.conv_reduce.weight',
                   'blocks.3.1.se.conv_expand.weight']
    sparsity_ = []
    for k, w in model.named_parameters():
        if k in weight_list:
            # self.mask_dict[k].apply_mask(w)
            sparsity = (w.numel() - w.nonzero().size(0)) / w.numel()
            if print:
                print('{} with sparsity {}'.format(k, sparsity))
            sparsity_.append(sparsity)
    return np.mean(sparsity_)

