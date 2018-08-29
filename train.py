import argparse
from collections import Counter, defaultdict
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import QuickThoughtsModel


class Corpus(Dataset):
    def __init__(self, max_length=30, unk_threshold=5):
        self.word_index = defaultdict(lambda: len(self.word_index))
        self.max_length = max_length
        self.unk_threshold = unk_threshold

    @staticmethod
    def read_corpus(corpus_file):
        self = Corpus()
        self.dataset = []

        self.word_index['EOS']  # 0
        self.word_index['UNK']  # 1

        with open(corpus_file, 'r', encoding='utf8') as f:
            sentences = [line.split() for line in f if line]
            counter = Counter(chain.from_iterable(sentences))
            dataset = []

            for sent in sentences:
                indices = []
                for word in sent[:self.max_length]:
                    if counter[word] <= self.unk_threshold:
                        word = 'UNK'
                    self.word_index[word]
                    indices.append(self.word_index[word])

                if len(indices) < self.max_length:
                    indices += ([self.word_index['EOS']] *
                                (self.max_length - len(indices)))
                dataset.append(indices)

        self.index_word = {v: k for k, v in self.word_index.items()}
        self.dataset = torch.LongTensor(dataset)

        return self

    def vocab_count(self):
        return len(self.word_index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def train(model, dataset, args, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=False)

    for epoch in range(1, args.epoch + 1):
        for i, batch in enumerate(train_loader):
            loss = model(batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), args.save)


def parse_args():
    epoch = 10
    batch_size = 256
    embed_size = 100
    thought_size = 2400
    context_size = 1
    dropout=0.
    bidirectional = False
    pretrained_weight = None

    parser = argparse.ArgumentParser(description='Quick-Thought Vectors')

    parser.add_argument('--train', type=str, required=True,
                        help='source corpus file')
    parser.add_argument('--save', type=str, required=True,
                        help='path to save the final model')
    parser.add_argument('--cuda', type=int, required=True,
                        help='set it to 1 for running on GPU, 0 for CPU')
    parser.add_argument('--epoch', '-e', default=epoch, metavar='N', type=int,
                        help=f'number of training epochs (default: {epoch})')
    parser.add_argument('--batch_size', '-b', default=batch_size,
                        metavar='N', type=int,
                        help=f'minibatch size for training (default: {batch_size})')
    parser.add_argument('--wembed', '-w', default=embed_size,
                        metavar='N', type=int,
                        help=f'the dimension of word embedding (default: {embed_size})')
    parser.add_argument('--sembed', '-s', default=thought_size,
                        metavar='N', type=int,
                        help=f'the dimension of sentence embedding (default: {thought_size})')
    parser.add_argument('--context', '-c', default=context_size,
                        metavar='N', type=int,
                        help=f'predict previous and next N sentences (default: {context_size})')
    parser.add_argument('--dropout', '-d', default=dropout,
                        metavar='N', type=int,
                        help=f'dropout rate (default: {dropout})')
    parser.add_argument('--bidirectional', default=bidirectional,
                        type=bool, choices=[True, False],
                        help=f'use bi-directional model (default: {bidirectional})')
    parser.add_argument('--pretrained', default=pretrained_weight, type=str,
                        help='pre-trained word embeddings file')
    parser.add_argument('--seed', type=int, default='1234', help='random seed')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    dataset = Corpus.read_corpus(args.train)
    vocab_size = dataset.vocab_count()

    model = QuickThoughtsModel(vocab_size, args.wembed, args.sembed,
                               args.context, dropout=args.dropout,
                               bidirectional=args.bidirectional,
                               pretrained_weight=args.pretrained,
                               device=device)
    model.to(device)
    print(model)

    train(model, dataset, args, device)


if __name__ == '__main__':
    main()
