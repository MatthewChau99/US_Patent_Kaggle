import os

import pandas as pd
from transformers import BertTokenizer

from src.config import ROOT_PATH

train_df = pd.read_csv(os.path.join(ROOT_PATH, 'data', 'train.csv'))
test_df = pd.read_csv(os.path.join(ROOT_PATH, 'data', 'test.csv'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def context_to_text(context: str) -> str:
    """
    Takes a context string and converts it to the corresponding classification
    :param context: strings in df['context']. e.g. "A47"
    :return: the name for the context code
    """
    patent_clf_map = {
        'A': 'Human Necessities',
        'B': 'Operations and Transport',
        'C': 'Chemistry and Metallurgy',
        'D': 'Textiles',
        'E': 'Fixed Constructions',
        'F': 'Mechanical Engineering',
        'G': 'Physics',
        'H': 'Electricity',
        'Y': 'Emerging Cross-Sectional Technologies'
    }

    if context[0] not in patent_clf_map:
        print(f'{context[0]} is not in the dict')
        return ''

    return patent_clf_map[context[0]]


def concat(*args) -> str:
    """
    Join the strings together for inputting into transformer
    :param args: strings being concatenated
    :return: concatenated string using # as separator
    """
    return ' # '.join(args)


def preprocess(file_name, dataframe=train_df) -> pd.DataFrame:
    """
    This preprocessor does three things to the train_df in /data:
    1. Converts column "context" into its corresponding names for the context codes in column "context_text"
    2. Concatenates the "anchor", "target" and "context_text" together into a single string in column "concat".
    3. Tokenizes the concatenated string into vector in column "concat_vec".
    After preprocessing, it outputs to a file in /data/processed.
    :param file_name: the output file name
    :return: the processed Dataframe
    """
    
    dataframe['context_text'] = dataframe['context'].apply(context_to_text)
    dataframe['concat'] = dataframe.apply(lambda x: concat(x['anchor'], x['target'], x['context_text']), axis=1)
    dataframe['concat_vec'] = dataframe['concat'].apply(
        lambda x: tokenizer(x, padding='longest', truncation=True)['input_ids'])

    if not os.path.exists(os.path.join(ROOT_PATH, 'data', 'processed')):
        os.mkdir(os.path.join(ROOT_PATH, 'data', 'processed'))

    dataframe.to_csv(os.path.join(ROOT_PATH, 'data', 'processed', file_name))
    return dataframe


if __name__ == '__main__':
    preprocess(file_name='test.csv', dataframe=test_df)
