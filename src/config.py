from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parent.parent
train_path = 'data/train.csv'
test_path = 'data/test.csv'
titles_path = 'data/titles.csv'
model_path = ''
ensemble_paths = ['../input/us-patent-deberta-cv2/cv_2', '../input/us-patent-deberta-cv4/cv_3']
ensemble_5folds = ['../input/deberta-v3-5folds/', '../input/bert-for-patent-5fold/', '../input/deberta-large-v1/', '../input/xlm-roberta-large-5folds/']

batch_size = 16
max_len = 133

