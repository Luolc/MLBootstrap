from mlbootstrap.base import MlbtsBaseModule
import os


class AbstractModel(MlbtsBaseModule):
    def __init__(self, name: str = 'abstract_model'):
        self.name = name
        self.dataset = None

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def set_dataset(self, dataset):
        self.dataset = dataset

    def _build_graph(self, mode: str):
        pass

    def training_parameter(self, name: str):
        return self._config['train'][name]

    def set_visible_gpus(self):
        visible_gpus = self._config.get('gpu', None)
        if visible_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus


class BasicModel(AbstractModel):
    def __init__(self, name: str = 'basic_model'):
        super(BasicModel, self).__init__(name)

    def train(self):
        pass

    def evaluate(self):
        pass
