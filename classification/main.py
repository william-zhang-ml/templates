"""
Template classifier training script.
"""
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm.auto import tqdm


def build_instance(blueprint: DictConfig, updates: Dict = None) -> Any:
    """Dynamically-build an arbitrary class instance.

    Args:
        blueprint (DictConfig): config w/keys module, class_name, kwargs
        updates (Dict): additional/non-default kwargs

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


def do_eval_pass(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criteria: Callable,
    metric: Callable = None
) -> Tuple[Any, Any]:
    """Run input data through the model.

    Args:
        loader (torch.utils.data.DataLoader): validation batch loader
        model (torch.nn.Module): neural network to train/validate
        criteria (Callable): task loss function
        metric (Callable): task metric

    Returns:
        Tuple[Any, Any]: results of calling criteria and metric
    """
    device = next(model.parameters()).device
    num_batches = len(loader)
    outp = []
    labels = []
    with torch.no_grad():
        progbar = tqdm(loader, leave=False)
        for i_batch, (inp, target) in enumerate(progbar):
            outp.append(model(inp.to(device)).cpu())
            labels.append(target)
            progbar.set_postfix({
                'batch': f'{i_batch + 1}/{num_batches}'
            })
    outp = torch.cat(outp)
    labels = torch.cat(labels)
    loss = criteria(outp, labels)
    metric_val = metric(outp, labels)
    return loss, metric_val


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig = None) -> None:
    """Train a classifier.

    Args:
        config (DictConfig): script configuration
    """
    outdir = Path(HydraConfig.get().runtime.output_dir)
    device = torch.device(config.device)
    train_board = SummaryWriter(log_dir=f'tensorboard/train-{outdir.stem}')
    valid_board = SummaryWriter(log_dir=f'tensorboard/valid-{outdir.stem}')

    # get dataset and batch loaders
    preprocess = Compose(
        [build_instance(blueprint) for blueprint in config.dataset.preprocess]
    )
    train_data = build_instance(
        config.dataset.train,
        {'transform': preprocess}
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_data = build_instance(
        config.dataset.valid,
        {'transform': preprocess}
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=config.batch_size,
        shuffle=False
    )

    # get model and optimizers
    model = build_instance(
        config.model,
        {'num_classes': len(train_data.classes)}
    ).to(device)
    criteria = build_instance(config.optimization.criteria)
    metric = get_top1_acc
    optimizer = build_instance(
        config.optimization.optimizer,
        {'params': model.parameters()}
    )
    if 'scheduler' in config.optimization:
        scheduler = build_instance(
            config.optimization.scheduler,
            {'optimizer': optimizer}
        )

    # main training loop
    progbar = tqdm(range(config.num_epochs))
    num_batches = len(train_loader)
    step = 0
    valid_loss, valid_metric_val = float('nan'), float('nan')
    for i_epoch in progbar:
        # per-epoch loop
        for i_batch, (imgs, labels) in enumerate(train_loader):
            step += 1

            # gradient descent
            with torch.autocast(device_type=device.type):
                loss, metric_val = do_forward_pass(
                    imgs.to(device),
                    labels.to(device),
                    model,
                    criteria,
                    metric
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            progbar.set_postfix({
                'batch': f'{i_batch + 1}/{num_batches}',
                'loss': f'{loss:.03f}',
                'metric': f'{metric_val:.01f}',
                'valid loss': f'{valid_loss:.03f}',
                'valid metric': f'{valid_metric_val:.01f}'
            })
            train_board.add_scalar('loss', loss, step)
            train_board.add_scalar('metric', metric_val, step)

        # per-epoch updates
        if 'scheduler' in locals():
            scheduler.step()
        if i_epoch % config.epochs_per_valid == 0:
            valid_loss, valid_metric_val = do_eval_pass(
                valid_loader,
                model,
                criteria,
                metric
            )
            valid_board.add_scalar('loss', valid_loss, step)
            valid_board.add_scalar('metric', valid_metric_val, step)

    # save final model
    torch.onnx.export(
        model,
        imgs.to(device),
        outdir / 'final.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 1: 'row', 2: 'col'}
        }
    )


if __name__ == '__main__':
    main()
