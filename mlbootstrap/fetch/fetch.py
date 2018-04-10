import os
from ..base import MlbtsBaseModule
from pathlib import Path


class AbstractFetcher(MlbtsBaseModule):
    def finished(self):
        return False

    def fetch(self):
        raise NotImplementedError

    def check(self):
        def __check_node(node, path):
            path = os.path.join(path, node['path'])
            if 'task' in node:
                self._check_dataset(path)
            if 'sub' in node:
                for _i in node['sub']:
                    __check_node(_i, path)

        for i in self._config['dataset']:
            __check_node(i, '')

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
