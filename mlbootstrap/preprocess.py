from mlbootstrap.base import MlbtsBaseModule
import os
from pathlib import Path


class AbstractPreprocessor(MlbtsBaseModule):
    def finished(self):
        return False

    def process(self):
        self._on_start()
        for node in self._datasets:
            self._on_next(**node)
        self._on_complete()

    def _on_start(self):
        pass

    def _on_next(self, src, dst, task):
        pass

    def _on_complete(self):
        pass

    def load_processed(self):
        self.meta = self._load_meta_info()
        self.dataset = self._load_dataset()

    def _load_meta_info(self):
        raise NotImplementedError

    def _load_dataset(self):
        raise NotImplementedError

    def check(self):
        for node in self._datasets:
            self._check_processed(node['task'])

    def _check_processed(self, task):
        raise NotImplementedError


class BasicPreprocessor(AbstractPreprocessor):
    def finished(self):
        for node in self._datasets:
            if not Path(node['dst']).is_dir():
                return False
        return True

    def _load_meta_info(self):
        return None

    def _load_dataset(self):
        return None

    def _check_processed(self, task):
        path = os.path.join(self._config['processed']['path'], task)
        if not Path(path).is_dir():
            raise FileNotFoundError("Processed task '{}' does not exist".format(task))
