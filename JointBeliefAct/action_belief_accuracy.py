import torch

def obtain_TP_TN_FN_FP(pred, act, TP, TN, FN, FP, elem_wise=False):
    if isinstance(pred, torch.Tensor):
        if elem_wise:
            TP += ((pred.data == 1) & (act.data == 1)).sum(0)
            TN += ((pred.data == 0) & (act.data == 0)).sum(0)
            FN += ((pred.data == 0) & (act.data == 1)).sum(0)
            FP += ((pred.data == 1) & (act.data == 0)).sum(0)
        else:
            TP += ((pred.data == 1) & (act.data == 1)).cpu().sum().item()
            TN += ((pred.data == 0) & (act.data == 0)).cpu().sum().item()
            FN += ((pred.data == 0) & (act.data == 1)).cpu().sum().item()
            FP += ((pred.data == 1) & (act.data == 0)).cpu().sum().item()
        return TP, TN, FN, FP
    else:
        TP += ((pred > 0).astype('long') & (act > 0).astype('long')).sum()
        TN += ((pred == 0).astype('long') & (act == 0).astype('long')).sum()
        FN += ((pred == 0).astype('long') & (act > 0).astype('long')).sum()
        FP += ((pred > 0).astype('long') & (act == 0).astype('long')).sum()
        return TP, TN, FN, FP

