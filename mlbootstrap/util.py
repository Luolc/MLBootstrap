import os
from itertools import chain


def dataset_tree_as_list(config):
    def __gen(node, path=''):
        path = os.path.join(path, node['path'])
        if 'task' in node:
            task = node['task']
            yield dict(src=path, dst=os.path.join(config['processed']['path'], task), task=task)
        if 'sub' in node:
            for _i in node['sub']:
                for _item in __gen(_i, path):
                    yield _item

    return chain(*[__gen(i) for i in config['dataset']])
