from ..base import MlbtsBaseModule
from ..util.config import dataset_tree_as_list
import os
from pathlib import Path


class AbstractPreprocessor(MlbtsBaseModule):
    def finished(self):
        return False

    def process(self):
        self._on_start()
        for node in dataset_tree_as_list(self._config):
            self._on_next(node['src'], node['dst'], node['task'])
        self._on_complete()

    def _on_start(self):
        pass

    def _on_next(self, src, dst, task):
        pass

    def _on_complete(self):
        pass

    def load_cache(self):
        self.meta = self._load_meta_info()
        self.dataset = self._load_dataset()

    def _load_meta_info(self):
        raise NotImplementedError

    def _load_dataset(self):
        raise NotImplementedError

    def check(self):
        for node in dataset_tree_as_list(self._config):
            self._check_cache(node['task'])

    def _check_cache(self, task):
        raise NotImplementedError


class BasicPreprocessor(AbstractPreprocessor):
    def finished(self):
        for node in dataset_tree_as_list(self._config):
            if not Path(node['dst']).is_dir():
                return False
        return True

    def _load_meta_info(self):
        return None

    def _load_dataset(self):
        return None

    def _check_cache(self, task):
        path = os.path.join(self._config['cache']['path'], task)
        if not Path(path).is_dir():
            raise FileNotFoundError("Processed task '{}' does not exist".format(task))
