from mlbootstrap.preprocess import BasicPreprocessor
import os
import shutil
import json
import numpy as np
import pandas as pd
import tensorflow as tf


class DataPreprocessor(BasicPreprocessor):
    _VOCAB_FILENAME = 'vocab_processor.obj'
    _DATASET_META = 'meta.json'

    def _on_next(self, src, dst, task):
        shutil.rmtree(dst)
        os.makedirs(dst)

        stop_words = {'a', 'an', 'the'}
        raw_data = {}
        for filename in os.listdir(src):
            def __parse_insight(fp):
                with open(fp, 'r') as f:
                    result = json.load(f)
                    for item in result:
                        item['title'] = item['title'].replace('|', ' ').lower()
                        item['title'] = ' '.join(
                            w for w in item['title'].split() if w not in stop_words)
                    return result

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

        # merge insights
        # insights = [{**i, 'company': k} for k, v in raw_data.items() for i in v['insights']]
        # insights = [[{**i, 'company': k} for i in v['insights']] for k, v in raw_data.items()]

        companies = list(raw_data.keys())
        train_sep = round(len(companies) * 0.6)
        val_sep = round(len(companies) * 0.8)

        max_doc_length = 0
        for k, v in raw_data.items():
            samples = pd.DataFrame()
            insights = v['insights']
            titles = [i['title'] for i in insights]
            max_doc_length = max((len(t.split()) for t in titles))
            samples['title'] = titles
            samples['significant'] = [i['sign'] for i in insights]
            samples['target'] = [i['prob'] for i in insights]
            samples = samples.sample(frac=1).reset_index(drop=True)  # random shuffle
            samples.to_csv(os.path.join(dst, '{}.csv'.format(k)))

        # process vocabulary
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc_length)

        # save vocabulary
        train_titles = [i['title'] for k, v in raw_data.items() if k in companies[:train_sep]
                        for i in v['insights']]
        vocab_processor.fit(train_titles)
        vocab_processor.save(os.path.join(dst, self._VOCAB_FILENAME))

        with open(os.path.join(dst, self._DATASET_META), 'w') as meta_f:
            json.dump({
                'train': companies[:train_sep],
                'val': companies[train_sep:val_sep],
                'test': companies[val_sep:],
                'max_doc_length': max_doc_length,
                'vocab': len(vocab_processor.vocabulary_),
            }, meta_f)

    def _load_dataset(self):
        result = {}

        task = self._config['train']['task']
        dst = next(node['dst'] for node in self._datasets if node['task'] == task)
        vocab_path = os.path.join(dst, self._VOCAB_FILENAME)
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)

        with open(os.path.join(dst, self._DATASET_META), 'r') as meta_f:
            meta_data = json.load(meta_f)

        def __get_batches(companies):
            __ret = []
            for company_name in companies:
                samples = pd.read_csv(os.path.join(dst, '{}.csv'.format(company_name)))
                batch = {'title': np.array(list(vocab_processor.transform(samples['title']))),
                         'significant': np.array(samples['significant']),
                         'target': np.array(samples['target']),
                         'batch_size': len(samples),
                         }
                __ret.append(batch)
            return __ret

        result['train'] = __get_batches(meta_data['train'])
        result['val'] = __get_batches(meta_data['val'])
        result['test'] = __get_batches(meta_data['test'])
        result['max_doc_length'] = meta_data['max_doc_length']
        result['vocab'] = meta_data['vocab']

        return result
