import yaml
from preprocessor import BasePreprocessor


with open('config.yaml', 'r') as stream:
    try:
        print(yaml.load(stream))
    except yaml.YAMLError as ex:
        print(ex)
