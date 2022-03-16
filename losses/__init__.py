from .F1Loss import F1Loss
from .FocalLoss import FocalLoss
from .BCELogLoss import BCELogLoss
from .BCELogLossV2 import BCELogLossV2
from .FCLoss import BFocalLoss
import torch.nn as nn

# class L1Loss(nn.L1Loss, base.Loss):
#     pass


# class MSELoss(nn.MSELoss, base.Loss):
#     pass


class CrossEntropyLoss(nn.CrossEntropyLoss):
    pass


# class NLLLoss(nn.NLLLoss, base.Loss):
#     pass


# class NLLLoss2d(nn.NLLLoss2d, base.Loss):
#     pass


# class BCELoss(nn.BCELoss, base.Loss):
#     pass


# class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
#     pass


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import torch.nn as nn
# from .DiceLoss import DiceLoss
# from .F1Loss import F1Loss
# from .FocalLoss import FocalLoss
# from .BCELogLoss import BCELogLoss
# from .BCELogLossV2 import BCELogLossV2
# from .FCLoss import BFocalLoss
# # from .SCELoss import SCELoss
# from .FusedJSD import JsdCrossEntropy, LabelSmoothingCrossEntropy
# from .ClassBalancedLoss import CBalancedLoss

# __factory = {
#     'dice': DiceLoss,
#     'margin': nn.MultiLabelSoftMarginLoss,
#     'focal': FocalLoss,
#     'f1': F1Loss,
#     'bce': BCELogLoss,
#     'ce': nn.CrossEntropyLoss,
#     'bce_v2': BCELogLossV2,
#     'bfocal': BFocalLoss,
#     'jsd': JsdCrossEntropy,
#     'lsce': LabelSmoothingCrossEntropy,
#     'class_balance': CBalancedLoss
# }


# def init_loss_func(name, **kwargs):
#     avai_losses = list(__factory.keys())
#     if name not in avai_losses:
#         raise KeyError(
#             'Invalid loss function name. Received "{}", but expected to be one of {}'.format(name, avai_losses))
#     return __factory[name](**kwargs)
