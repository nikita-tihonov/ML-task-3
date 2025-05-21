import numpy as np
from torchtext.data import Field, Example, Dataset, BucketIterator
import pandas as pd
from tqdm.auto import tqdm
import torch


if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    DEVICE = torch.device('cuda')

else:
    from torch import FloatTensor, LongTensor
    DEVICE = torch.device('cpu')

np.random.seed(42)

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
fields = [('source', word_field), ('target', word_field)]
data = pd.read_csv('events.csv', delimiter=',')
examples = []
for _, row in tqdm(data.iterrows(), total=len(data)):
    source_text = word_field.preprocess(row.text)
    target_text = word_field.preprocess(row.title)
    examples.append(Example.fromlist([source_text, target_text], fields))

dataset = Dataset(examples, fields)
train_dataset, val_dataset = dataset.split(split_ratio=0.85)
train_dataset, test_dataset = train_dataset.split(split_ratio=0.9)
print('Train size =', len(train_dataset))
print('Test size =', len(test_dataset))
print('Val size =', len(val_dataset))
word_field.build_vocab(train_dataset, min_freq=7)
print('Vocab size =', len(word_field.vocab))



train_iter, test_iter = BucketIterator.splits(datasets=(train_dataset, test_dataset), batch_sizes=(16, 32),
                                              shuffle=True, device=DEVICE, sort=False)