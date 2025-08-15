def topk_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1)
        correct_k = correct_k.float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res