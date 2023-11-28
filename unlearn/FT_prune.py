import copy
import pruner
import trainer
from .FT import FT_l1, FT


def FT_prune(data_loaders, model, criterion, args):
    test_loader = data_loaders["test"]

    # save checkpoint
    initialization = copy.deepcopy(model.state_dict())

    # unlearn
    FT_l1(data_loaders, model, criterion, args)

    # val
    pruner.check_sparsity(model)
    trainer.validate(test_loader, model, criterion, args)

    return model
