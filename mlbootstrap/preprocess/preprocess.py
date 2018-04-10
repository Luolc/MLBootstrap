from ..base import MlbtsBaseModule
import os
from pathlib import Path


class AbstractPreprocessor(MlbtsBaseModule):
    def finished(self):
        return False

    def process(self):
        raise NotImplementedError

    def load_cache(self):
        self.meta = self._load_meta_info()
        self.dataset = self._load_dataset()

    def _load_meta_info(self):
        raise NotImplementedError

    def _load_dataset(self):
        raise NotImplementedError

    def check(self):
        def __check_node(node):
            if 'task' in node:
                self._check_cache(node['task'])
            if 'sub' in node:
                for _i in node['sub']:
                    __check_node(_i)

        for i in self._config['dataset']:
            __check_node(i)

    def _check_cache(self, task):
        raise NotImplementedError


class BasicPreprocessor(AbstractPreprocessor):
    def finished(self):
        return True

    def process(self):
        pass

    def _load_meta_info(self):
        return None

    def _load_dataset(self):
        return None

    def _check_cache(self, task):
        path = os.path.join(self._config['cache']['path'], task)
        if not Path(path).is_dir():
            raise FileNotFoundError("Processed task '{}' does not exist".format(task))
