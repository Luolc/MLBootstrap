from ..base import MlbtsBaseModule
from ..util.config import dataset_tree_as_list
from pathlib import Path


class AbstractFetcher(MlbtsBaseModule):
    def finished(self):
        return False

    def fetch(self):
        raise NotImplementedError

    def check(self):
        for node in dataset_tree_as_list(self._config):
            self._check_dataset(node['src'])

    def _check_dataset(self, path):
        raise NotImplementedError


class BasicFetcher(AbstractFetcher):
    def finished(self):
        return True

    def fetch(self):
        pass

    def _check_dataset(self, path):
        if not Path(path).is_dir():
            raise FileNotFoundError("Dataset '{}' does not exist".format(path))
