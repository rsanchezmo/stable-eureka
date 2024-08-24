from typing import Dict, Union, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
from stable_eureka import StableEureka


class EurekaSignature(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the Eureka component.

        This method should handle any necessary setup or configuration, including any possible callback
        objects to keep track of the metrics of the training or evaluation process
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Execute the main functionality of the Eureka component.

        This method should implement the core logic, whether it's
        training, evaluation, or any other primary operation.
        """
        pass


class FlexibleStableEureka:
    """
    This class provides a customizable interface to the StableEureka optimization process,
    allowing for easy modification of key components such as the trainer, evaluator,
    environment directory, and main environment file.

    :param config_path: Path to the configuration file for StableEureka.
    :param trainer: Custom trainer class implementing the EurekaSignature interface.
                    If None, the default StableEureka trainer will be used.
    :param evaluator: Custom evaluator class implementing the EurekaSignature interface.
                      If None, the default StableEureka evaluator will be used.
    :param env_dir: Directory containing the environment files.
                    If None, the default StableEureka environment directory will be used.
    :param main_env_file: Name of the main environment file.
                          If None, the default StableEureka main environment file will be used.
    """

    def __init__(self, config_path: Union[str, Path], trainer: Optional[EurekaSignature] = None,
                 evaluator: Optional[EurekaSignature] = None, env_dir: Optional[str] = None,
                 main_env_file: Optional[str] = None):

        self._config_path = Path(config_path)
        self._stable_eureka = StableEureka(config_path=self._config_path)

        self._modifiable_attrs = {
            'trainer': trainer,
            'evaluator': evaluator,
            'env_dir': env_dir,
            'main_env_file': main_env_file
        }

        for attr, value in self._modifiable_attrs.items():
            if value is not None:
                self._set_attribute(attr, value)

    def _set_attribute(self, attr: str, value: Any):
        if attr in self._modifiable_attrs:
            if hasattr(self._stable_eureka, attr):
                if attr in ['trainer', 'evaluator'] and not isinstance(value, EurekaSignature):
                    raise TypeError(f"{attr.capitalize()} must be an instance of EurekaSignature")
                setattr(self._stable_eureka, attr, value)
            else:
                raise AttributeError(f"StableEureka instance has no attribute '{attr}'")
        else:
            raise ValueError(f"Attribute '{attr}' is not modifiable")

    @property
    def trainer(self) -> Optional[EurekaSignature]:
        return getattr(self._stable_eureka, 'trainer', None)

    @trainer.setter
    def trainer(self, value: Optional[EurekaSignature]):
        self._set_attribute('trainer', value)

    @property
    def evaluator(self) -> Optional[EurekaSignature]:
        return getattr(self._stable_eureka, 'evaluator', None)

    @evaluator.setter
    def evaluator(self, value: Optional[EurekaSignature]):
        self._set_attribute('evaluator', value)

    @property
    def env_dir(self) -> Optional[str]:
        return getattr(self._stable_eureka, 'env_dir', None)

    @env_dir.setter
    def env_dir(self, value: Optional[str]):
        if value is not None:
            value = Path(value)
        self._set_attribute('env_dir', value)

    @property
    def main_env_file(self) -> Optional[str]:
        return getattr(self._stable_eureka, 'main_env_file', None)

    @main_env_file.setter
    def main_env_file(self, value: Optional[str]):
        self._set_attribute('main_env_file', value)

    def get_attributes(self) -> Dict[str, Any]:
        return {attr: getattr(self._stable_eureka, attr)
                for attr in self._modifiable_attrs
                if hasattr(self._stable_eureka, attr)}

    def __getattr__(self, name):
        return getattr(self._stable_eureka, name)

