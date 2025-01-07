from .sgd import SGD_Simple
from .svrg import SVRG_k, SVRG_Snapshot
def initialize_optimizer(args, model, model_snapshot):
    """ optimizer initialization function, if optimizer is SVRG, then initialize the snapshot optimizer.

    Args:
        args (_type_):
        model (nn.Module): _description_
        model_snapshot (_type_): _description_

    Returns:
        _type_: _description_
    """
    optimizers = {
        "SGD": SGD_Simple,
        "SVRG": SVRG_k
    }
    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters()) if args.optimizer == 'SVRG' else None
    return optimizer, optimizer_snapshot