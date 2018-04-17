from mlbootstrap.base import MlbtsBaseModule


class AbstractModel(MlbtsBaseModule):
    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class BasicModel(AbstractModel):
    def train(self):
        pass

    def evaluate(self):
        pass
