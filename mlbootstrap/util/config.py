import os
from itertools import chain


def dataset_tree_as_list(config):
    def __gen(node, path=''):
        path = os.path.join(path, node['path'])
        if 'task' in node:
            task = node['task']
            yield {
                'src': path,
                'dst': os.path.join(config['cache']['path'], task),
                'task': task,
            }
        if 'sub' in node:
            for _i in node['sub']:
                __gen(_i, path)

    return chain(*[__gen(i) for i in config['dataset']])
