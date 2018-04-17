from mlbootstrap.preprocess import BasicPreprocessor
from pathlib import Path
import os
import shutil
import json
import numpy as np
import pandas as pd
import tflearn

_TABLE_INSIGHT_DIR = 'table-insight'
_VOCAB_FILENAME = 'vocab_processor.obj'
_DATASET_META = 'meta.json'

_STOP_WORDS = {' ', 'a', 'an', 'the', 'and'}
_TYPE2ID = {'rate': 0, 'trend': 1}


def _parse_insight(fp):
    with open(fp, 'r') as f:
        result = json.load(f)
        for item in result:
            item['title'] = ' '.join(
                w.lower() for w in item['title'] if w and w not in _STOP_WORDS)
        return [item for item in result if item['title']]


def _parse_table(fp):
    return pd.read_csv(fp).as_matrix()


class DataPreprocessor(BasicPreprocessor):
    def _on_next(self, src, dst, task):
        # if Path(dst).exists():
        #     shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)

        raw_data = {}
        for filename in os.listdir(os.path.join(src, _TABLE_INSIGHT_DIR)):
            file_path = os.path.join(src, _TABLE_INSIGHT_DIR + '/' + filename)
            table_id = filename.replace('.txt', '').replace('.csv', '')
            if table_id not in raw_data:
                raw_data[table_id] = {}
            if filename.endswith('.txt'):
                raw_data[table_id]['insights'] = _parse_insight(file_path)
            if filename.endswith('.csv'):
                raw_data[table_id]['table'] = _parse_table(file_path)

        table_insight_dir = os.path.join(dst, _TABLE_INSIGHT_DIR)
        os.makedirs(table_insight_dir, exist_ok=True)
        max_doc_length = 0
        i = 0
        for k, v in list(raw_data.items()):
            i += 1
            if i % 1000 == 0:
                print('processed {} insights'.format(i))

            out_path = os.path.join(table_insight_dir, '{}.csv'.format(k))
            if Path(out_path).exists():
                continue
            if ('insights' not in v) or len(v['insights']) < 5:
                del raw_data[k]
                continue

            samples = pd.DataFrame()
            insights = v['insights']
            titles = [i['title'] for i in insights]
            max_doc_length = max((len(t.split()) for t in titles))
            samples['title'] = titles
            samples['significant'] = [i['sig'] for i in insights]
            samples['type'] = [_TYPE2ID[i['type']] for i in insights]
            samples['target'] = [i['prob'] for i in insights]
            samples = samples.sample(frac=1).reset_index(drop=True)  # random shuffle
            samples.to_csv(out_path)

        # process vocabulary
        min_frequency = self._config['processed']['vocab']['min_frequency']
        vocab_processor = tflearn.data_utils.VocabularyProcessor(
            max_doc_length, min_frequency=min_frequency)

        table_ids = list(raw_data.keys())
        train_sep = round(len(table_ids) * 0.6)
        val_sep = round(len(table_ids) * 0.8)

        # save vocabulary
        train_titles = [i['title'] for k, v in raw_data.items() if k in table_ids[:train_sep]
                        for i in v['insights']]
        vocab_processor.fit(train_titles)
        vocab_processor.save(os.path.join(dst, _VOCAB_FILENAME))

        with open(os.path.join(dst, _DATASET_META), 'w') as meta_f:
            json.dump({
                'train': table_ids[:train_sep],
                'val': table_ids[train_sep:val_sep],
                'test': table_ids[val_sep:],
                'max_doc_length': max_doc_length,
                'vocab_size': len(vocab_processor.vocabulary_),
            }, meta_f)

    def _load_dataset(self):
        result = {}

        task = self._config['train']['task']
        dst = next(node['dst'] for node in self._datasets if node['task'] == task)
        vocab_path = os.path.join(dst, _VOCAB_FILENAME)
        vocab_processor = tflearn.data_utils.VocabularyProcessor.restore(vocab_path)

        with open(os.path.join(dst, _DATASET_META), 'r') as meta_f:
            meta_data = json.load(meta_f)

        def __get_batches(table_ids):
            __ret = []
            table_insight_dir = os.path.join(dst, _TABLE_INSIGHT_DIR)
            for table_id in table_ids:
                samples = pd.read_csv(os.path.join(table_insight_dir, '{}.csv'.format(table_id)))
                batch = {
                    'title': np.array(list(vocab_processor.transform(samples['title']))),
                    'significant': np.array(samples['significant']),
                    'type': np.array(samples['type']),
                    'target': np.array(samples['target']),
                    'batch_size': len(samples),
                }
                __ret.append(batch)
            return __ret

        result['train'] = __get_batches(meta_data['train'])
        result['val'] = __get_batches(meta_data['val'])
        result['test'] = __get_batches(meta_data['test'])
        result['max_doc_length'] = meta_data['max_doc_length']
        result['vocab_size'] = meta_data['vocab_size']

        return result
