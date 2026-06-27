import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy
# import lorentz as L

'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''













def contrastive_loss(m_skel, m_txt, tmp, target, reduction='mean'):
    # print("================")
    if len(m_skel.shape) > 2:
        logits_per_skel_txt = (tmp * m_skel * m_txt).sum(-1)
        loss = F.binary_cross_entropy_with_logits(logits_per_skel_txt, target.float(), reduction=reduction)
    else:  
        logits_per_skel_txt = tmp * (m_skel @ m_txt.t())
        logits_per_txt_skel = tmp * (m_txt @ m_skel.t())
        # print(logits_per_skel_txt.shape)
        loss = (F.cross_entropy(logits_per_skel_txt, target, reduction=reduction) + \
                F.cross_entropy(logits_per_txt_skel, target, reduction=reduction))

        # print(loss)        
    return loss, logits_per_skel_txt



def GAP_contrastive_loss(m_skel, m_txt, tmp, target, reduction='mean'):
    # print("================")
   
    logits_per_skel_txt = tmp * (m_skel @ m_txt.t())
    logits_per_txt_skel = tmp * (m_txt @ m_skel.t())
    
    label_g = gen_label(target)
    ground_truth = torch.tensor(label_g, dtype=m_skel.dtype, device=m_skel.device)


    loss_cl=KLLoss()

    loss_imgs = loss_cl(logits_per_skel_txt, ground_truth)
    loss_texts = loss_cl(logits_per_txt_skel, ground_truth)
    loss=((loss_imgs + loss_texts) / 2)

      
    return loss, logits_per_skel_txt





class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss



def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt  



def euclidean_dist(x, y):
    # check shape:
    if len(list(x.size())) == 3: # n, p, d
        # part-base
        n = x.size(0)
        q = y.size(0)
        p = x.size(1)
        d = x.size(2)
        assert d == y.size(2)
        assert p == y.size(1)
        x = x.unsqueeze(1).expand(n, q, p, d)
        y = y.unsqueeze(0).expand(n, q, p, d)
        distance = torch.mean(torch.norm(x - y, dim=3), dim=2)

    elif len(list(x.size())) == 4: # n, t, p, d
        # part-base
        n = x.size(0)
        q = y.size(0)
        t = x.size(1)
        p = x.size(2)
        d = x.size(3)
        assert d == y.size(3)
        assert t == y.size(1)
        assert p == y.size(2)
        x = x.unsqueeze(1).expand(n, q, t, p, d)
        y = y.unsqueeze(0).expand(n, q, t, p, d)
        distance = torch.mean(torch.sum(torch.norm(x - y, dim=4), dim=3), dim=2)

    elif len(list(x.size())) == 5: # n, r, t, p, d
        # part-base
        n = x.size(0)
        q = y.size(0)
        r = x.size(1)
        t = x.size(2)
        p = x.size(3)
        d = x.size(4)
        assert d == y.size(4)
        assert t == y.size(2)
        assert p == y.size(3)

        x = x.unsqueeze(1).expand(n, q, r, t, p, d)
        y = y.unsqueeze(0).expand(n, q, r, t, p, d)
        distance = torch.mean(torch.sum(torch.sum(torch.norm(x - y, dim=5), dim=4), dim=3), dim=2)

    elif len(list(x.size())) == 2: # n, d
        # node-base
        n = x.size(0)
        q = y.size(0)
        d = x.size(1)
        assert d == y.size(1)
        x = x.unsqueeze(1).expand(n, q, d)
        y = y.unsqueeze(0).expand(n, q, d)
        distance = torch.norm(x - y, dim=2)
    
    return distance # (n, m)


def map_labels(query_labels, support_labels):
    
    # 确保输入是列表或numpy数组
    if torch.is_tensor(query_labels):
        query_labels = query_labels.cpu().numpy()
    if torch.is_tensor(support_labels):
        support_labels = support_labels.cpu().numpy()
    
    # 创建标签到索引的映射字典
    label_to_index = {label: idx for idx, label in enumerate(support_labels)}
    
    # 映射query标签
    mapped_labels = [label_to_index[label] for label in query_labels]
    
    return torch.tensor(mapped_labels)

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc, pred