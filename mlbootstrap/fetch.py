from mlbootstrap.base import MlbtsBaseModule
from pathlib import Path


class AbstractFetcher(MlbtsBaseModule):
    def finished(self):
        return False

    def fetch(self):
        raise NotImplementedError

    def check(self):
        for node in self._datasets:
            self._check_dataset(node['src'])

    def _check_dataset(self, path):
        raise NotImplementedError


class BasicFetcher(AbstractFetcher):
    def finished(self):
        for node in self._datasets:
            if not Path(node['src']).is_dir():
                return False
        return True

    def fetch(self):
        pass

    def _check_dataset(self, path):
        if not Path(path).is_dir():
            raise FileNotFoundError("Dataset '{}' does not exist".format(path))
