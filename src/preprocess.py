import pandas as pd
from transformers import BertTokenizer

from src.config import train_path, test_path, titles_path

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
titles_df = pd.read_csv(titles_path)
titles_map = pd.Series(titles_df.title.values, index=titles_df.code).to_dict()


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


def preprocess(dataframe=train_df) -> pd.DataFrame:
    """
    This preprocessor does three things to the train_df in /data:
    1. Converts column "context" into its corresponding names for the context codes in column "context_text"
    2. Concatenates the "anchor", "target" and "context_text" together into a single string in column "concat".
    3. Tokenizes the concatenated string into vector in column "concat_vec".
    After preprocessing, it outputs to a file in /data/processed.
    :param file_name: the output file name
    :return: the processed Dataframe
    """
    def code_to_title(code: str) -> str:
        if code not in titles_map:
            print(f'{code} is not in the dict')
            return ''

        return titles_map[code]

    def concat(*args) -> str:
        return ' [SEP] '.join(args)

    dataframe['context_text'] = dataframe['context'].apply(context_to_text)
    dataframe['title'] = dataframe['context'].apply(code_to_title)
    dataframe['concat'] = dataframe.apply(lambda x: concat(
        x['anchor'], x['target'], x['context_text'], x['title']), axis=1)

    return dataframe


if __name__ == '__main__':
    print(preprocess(dataframe=test_df))
