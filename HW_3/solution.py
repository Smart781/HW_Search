#!/usr/bin/env python
# ДЗ 3. Обучение ранжированию — шаблон решения.
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

def normalize_query_features(df, feature_cols):
    res = df.copy()
    for col in feature_cols:
        res[col] = df.groupby('QueryId')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    return res

def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    features = [column for column in data.columns if column not in ('QueryId', 'Label')]

    data = normalize_query_features(data, features)

    X = data[features]
    y = data['Label']
    qid = data['QueryId']

    train_pool = Pool(X, y, group_id=qid)

    model = CatBoostRanker(
        loss_function='YetiRank',
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=50,
        thread_count=-1
    )
    
    model.fit(train_pool)
    model.save_model(model_file)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    features = [column for column in test.columns if column not in ('ObjectId', 'QueryId')]

    test = normalize_query_features(test, features)
    model = CatBoostRanker()
    model.load_model(model_file)

    X_test = test[features]
    scores = model.predict(X_test)

    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    np.random.seed(42)

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
