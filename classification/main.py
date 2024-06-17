"""
Template classifier training script.
"""
import importlib
from typing import Any, Callable, Dict, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import Compose
from tqdm.auto import tqdm


def build_instance(blueprint: DictConfig, updates: Dict = None) -> Any:
    """Dynamically-build an arbitrary class instance.

    Args:
        blueprint (DictConfig): config w/keys module, class_name, kwargs
        updates (Dict): kwargs not specified in blueprint

    Returns:
        Any: instance defined by <module>.<class_name>(**kwargs)
    """
    module = importlib.import_module(blueprint.module)
    instance_kwargs = {}
    if 'kwargs' in blueprint:
        instance_kwargs.update(OmegaConf.to_container(blueprint.kwargs))
    if updates:
        instance_kwargs.update(updates)
    return getattr(module, blueprint.class_name)(**instance_kwargs)


def do_forward_pass(
    inp: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    criteria: Callable,
    metric: Callable = None
) -> Tuple[Any, Any]:
    """Run input data through the model.

    Args:
        inp (torch.Tensor): input data
        labels (torch.Tensor): labels or target values
        model (torch.nn.Module): neural network to train/validate
        criteria (Callable): task loss function
        metric (Callable): task metric

    Returns:
        Tuple[Any, Any]: results of calling criteria and metric
    """
    outp = model(inp)
    loss = criteria(outp, labels)
    metric_val = metric(outp, labels)
    return loss, metric_val


def get_top1_acc(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """Compute top-1 accuracy.

    Args:
        logits: network classification logits (batch_size, num_classes)
        labels: correct category lables (batch_size, )

    Returns:
        float: top-1 accuracy as a percent
    """
    assert logits.ndim == 2
    correct = logits.argmax(dim=-1) == labels
    return 100 * correct.sum() / correct.numel()


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig = None) -> None:
    """Train a classifier.

    Args:
        config (DictConfig): script configuration
    """
    device = torch.device(config.device)

    # get dataset and batch loaders
    preprocess = Compose(
        [build_instance(blueprint) for blueprint in config.dataset.preprocess]
    )
    train_data = build_instance(
        config.dataset.train,
        updates={'transform': preprocess}
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_data = build_instance(
        config.dataset.valid,
        updates={'transform': preprocess}
    )
    _ = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=config.batch_size,
        shuffle=False
    )

    # get model and optimizers
    model = build_instance(
        config.model,
        updates={'num_classes': len(train_data.classes)}
    ).to(device)
    criteria = build_instance(config.optimization.criteria)
    metric = get_top1_acc

    for _ in tqdm(range(config.num_epochs)):
        imgs, labels = next(iter(train_loader))
        with torch.autocast(device_type=device.type):
            loss, metric_val = do_forward_pass(
                imgs.to(device),
                labels.to(device),
                model,
                criteria,
                metric
            )


if __name__ == '__main__':
    main()
