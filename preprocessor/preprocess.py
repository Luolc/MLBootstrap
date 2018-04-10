import os


class BasePreprocessor:
    def __init__(self, config):
        self.config = config
        self.meta = None
        self.dataset = None

    def process(self):
        raise NotImplementedError

    def load_cache(self):
        self.meta = self.load_meta_info()
        self.dataset = self.load_dataset()

    def load_meta_info(self):
        raise NotImplementedError

    def load_dataset(self):
        raise NotImplementedError
