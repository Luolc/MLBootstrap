from mlbootstrap.util import dataset_tree_as_list


class MlbtsBaseModule:
    def set_config(self, config):
        self._config = config
        self._datasets = list(dataset_tree_as_list(self._config))
