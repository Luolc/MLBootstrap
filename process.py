from mlbootstrap.preprocess import BasicPreprocessor
import os
import json
import pandas as pd


class DataPreprocessor(BasicPreprocessor):
    def _on_next(self, src, dst, task):
        raw_data = {}
        for filename in os.listdir(src):
            def __parse_insight(fp):
                with open(fp, 'r') as f:
                    return json.load(f)

            def __parse_table(fp):
                return pd.read_csv(fp).as_matrix()[:, 1:5]

            file_path = os.path.join(src, filename)
            company = filename.replace('.txt', '').replace('.csv', '')
            if company not in raw_data:
                raw_data[company] = {}
            if filename.endswith('.txt'):
                raw_data[company]['insights'] = __parse_insight(file_path)
            if filename.endswith('.csv'):
                raw_data[company]['table'] = __parse_table(file_path)

        titles = [v['table'][:, 0].tolist() for v in raw_data.values()]
        titles = [t.split('|') for _titles in titles for t in _titles]
        titles = [t.split(' ') for _titles in titles for t in _titles]
        vocab = sorted(set(t.lower() for _titles in titles for t in _titles))




        os.makedirs(dst, exist_ok=True)
