from process import DataPreprocessor
from mlbootstrap import Bootstrap

bootstrap = Bootstrap('config.yaml', preprocessor=DataPreprocessor())
bootstrap.preprocess(force=True)
