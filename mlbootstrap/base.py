import os
from itertools import chain
from typing import List, Dict


class DatasetNode:
    def __init__(self, src: str, dst: str, task: str):
        self.src: str = src
        self.dst: str = dst
        self.task: str = task


def _dataset_tree_as_list(config: Dict):
    def __gen(node, path=''):
        path = os.path.join(path, node['path'])
        if 'task' in node:
            task = node['task']
            yield DatasetNode(
                src=path, dst=os.path.join(config['processed']['path'], task), task=task)
            # yield dict(src=path, dst=os.path.join(config['processed']['path'], task), task=task)
        if 'sub' in node:
            for _i in node['sub']:
                for _item in __gen(_i, path):
                    yield _item

    return chain(*[__gen(i) for i in config['dataset']])


class MlbtsBaseModule:
    def set_config(self, config: Dict):
        self._config: Dict = config
        self._datasets: List(DatasetNode) = list(_dataset_tree_as_list(self._config))

    def _get_dataset_node(self, task: str = None) -> DatasetNode:
        if not task:
            task = self._config['task']
        return next(node for node in self._datasets if node.task == task)

    def hyperparameter(self, name: str):
        return self._config['hyperparameter'][name]
