import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer

from config import max_len, model_path, test_path, titles_path, ensemble_paths, ensemble_5folds

# Reading in data
test_df = pd.read_csv(test_path)
titles_df = pd.read_csv(titles_path)
titles_map = pd.Series(titles_df.title.values, index=titles_df.code).to_dict()

# Loading tokenizers
single_tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizers = [AutoTokenizer.from_pretrained(model) for model in ensemble_paths]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, input='concat'):
        self.inputs = (df[input]).values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            **self.tokenizer(self.inputs[item], add_special_tokens=True, max_length=max_len)
        }


def model_predict(dataframe, model_path, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1).to(device)
    test_dataset = TestDataset(dataframe, tokenizer)

    trainer = Trainer(model, tokenizer=tokenizer)
    outputs = trainer.predict(test_dataset).predictions

    return outputs


def single_model_inference():
    scores = model_predict(test_df, model_path, single_tokenizer)

    submit_df = pd.DataFrame(
        np.c_[test_df['id'], scores], columns=['id',  'score'])
    submit_df.to_csv('submission.csv', index=False)

    return submit_df


def ensemble_inference(weights):
    scores = []

    for i, model_path in enumerate(ensemble_paths):
        print(f'Reading in model {model_path}')
        tokenizer = tokenizers[i]
        scores.append(model_predict(test_df, model_path, tokenizer))

    for i, model_path in enumerate(ensemble_5folds):
        fold_scores = []

        for fold in range(5):
            print(f'Reading in model {model_path} fold {fold}')
            tokenizer_5fold = AutoTokenizer.from_pretrained(
                f"{model_path}/fold{i}")
            fold_scores.append(model_predict(
                test_df, f"{model_path}/fold{i}", tokenizer_5fold))

        scores.append(np.mean(fold_scores, axis=0))

    final_scores = np.average(scores, axis=0, weights=weights)
    submit_df = pd.DataFrame(
        np.c_[test_df['id'], final_scores], columns=['id',  'score'])
    submit_df.to_csv('submission.csv', index=False)

    return submit_df
