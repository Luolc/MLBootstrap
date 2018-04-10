import yaml
from mlbootstrap.preprocess import AbstractPreprocessor
from mlbootstrap import Bootstrap

bootstrap = Bootstrap('config.yaml')
bootstrap.fetch()
