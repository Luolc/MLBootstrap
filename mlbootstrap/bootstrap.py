import yaml
from mlbootstrap.fetch import BasicFetcher
from mlbootstrap.preprocess import BasicPreprocessor


class Bootstrap:
    def __init__(self, config_path, fetcher=BasicFetcher(), preprocessor=BasicPreprocessor()):
        self.__fetcher = fetcher
        self.__preprocessor = preprocessor
        self.__load_config(config_path)

    def __load_config(self, config_path):
        with open(config_path, 'r') as stream:
            self.config = yaml.load(stream)

    def fetch(self):
        self.__fetcher.set_config(self.config)
        if not self.__fetcher.finished():
            self.__fetcher.fetch()
        self.__fetcher.check()

    def preprocess(self, force=False):
        self.fetch()

        self.__preprocessor.set_config(self.config)
        if (not self.__preprocessor.finished()) or force:
            self.__preprocessor.process()
        self.__preprocessor.load_cache()
        self.__preprocessor.check()
