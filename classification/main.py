"""
Template classifier training script.
"""
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig = None) -> None:
    """Train a classifier.

    Args:
        cfg (DictConfig): script configuration
    """
    for _ in tqdm(range(config.num_epochs)):
        pass


if __name__ == '__main__':
    main()
