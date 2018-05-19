from mlbootstrap.base import MlbtsBaseModule
from pathlib import Path


class AbstractFetcher(MlbtsBaseModule):
    def finished(self):
        return False

    def fetch(self):
        raise NotImplementedError

    def check(self):
        self._check_dataset(self._get_dataset_node().src)

    def _check_dataset(self, path: str):
        raise NotImplementedError


class BasicFetcher(AbstractFetcher):
    def finished(self) -> bool:
        return Path(self._get_dataset_node().src).is_dir()

    def fetch(self):
        pass

    def _check_dataset(self, path: str):
        if not Path(path).is_dir():
            raise FileNotFoundError("Dataset '{}' does not exist".format(path))
