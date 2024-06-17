"""
Template classifier training script.
"""
import importlib
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
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


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig = None) -> None:
    """Train a classifier.

    Args:
        cfg (DictConfig): script configuration
    """
    device = torch.device(config.device)

    # load dataset
    preprocess = Compose(
        build_instance(blueprint) for blueprint in config.dataset.preprocess
    )
    train_data = build_instance(
        config.dataset.train,
        updates={'transform': preprocess}
    )
    valid_data = build_instance(
        config.dataset.valid,
        updates={'transform': preprocess}
    )

    _ = build_instance(config.model).to(device)
    for _ in tqdm(range(config.num_epochs)):
        print(device.type)


if __name__ == '__main__':
    main()
