import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from preprocess import preprocess
from config import (batch_size, epochs, learning_rate, max_len, model_path, num_fold,
                    out_dir, titles_path, train_path, weight_decay)

df = pd.read_csv(train_path)
titles_df = pd.read_csv(titles_path)
titles_map = pd.Series(titles_df.title.values, index=titles_df.code).to_dict()
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, input='concat', output='score'):
        self.inputs = (df[input]).values
        self.scores = (df[output]).values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            **tokenizer(self.inputs[item], add_special_tokens=True, max_length=max_len),
            'labels': self.scores[item]
        }


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    metric = load_metric("pearsonr")
    computed = metric.compute(predictions=predictions, references=labels)

    return computed


def CV_train(train_df, val_df):
    train_dataset = TrainDataset(train_df, input='concat', output='score')
    val_dataset = TrainDataset(val_df, input='concat', output='score')

    args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        metric_for_best_model="pearsonr",
        load_best_model_at_end=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer, model


def train(data_df):
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    kfolds = np.array_split(data_df, num_fold)
    logs = []

    for i in tqdm(range(num_fold)):
        lis = []
        for j in range(num_fold):
            if j != i:
                lis.append(kfolds[j])
        train_df = pd.concat(lis)
        val_df = kfolds[i]

        trainer, model = CV_train(train_df, val_df)
        trainer.save_model(out_dir + '/cv_' + str(i))
        logs.append(trainer.state.log_history)

        del trainer
        del model

        with torch.no_grad():
            torch.cuda.empty_cache()

    import json
    with open(out_dir + '/manual_saved_logs.json', 'w') as json_file:
        json.dump(logs, json_file)


if __name__ == '__main__':
    train(preprocess(df))