from mlbootstrap.base import MlbtsBaseModule
import os
from pathlib import Path


class AbstractPreprocessor(MlbtsBaseModule):
    def finished(self) -> bool:
        return False

    def process(self):
        self._on_start()
        print("Start processing data for task '{}'.".format(self._get_dataset_node().task))
        self._on_next(**self._get_dataset_node().__dict__)
        self._on_complete()

    def _on_start(self):
        pass

    def _on_next(self, src: str, dst: str, task: str):
        pass

    def _on_complete(self):
        pass

    def load_processed(self):
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        raise NotImplementedError

    def check(self):
        if not Path(self._get_dataset_node().dst).is_dir():
            raise FileNotFoundError("Processed task '{}' does not exist".format(task))


class BasicPreprocessor(AbstractPreprocessor):
    def finished(self) -> bool:
        return Path(self._get_dataset_node().dst).is_dir()

    def _load_dataset(self):
        return None
