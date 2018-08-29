import numpy as np
import torch
import torch.nn as nn


class QuickThoughtsModel(nn.Module):
    def __init__(self, vocab_size, embed_size, thought_size, context_size,
                 dropout=0.3, bidirectional=False, pretrained_weight=None,
                 device=torch.device('cuda')):
        super().__init__()

        self.context_size = context_size
        self.encoder = Encoder(vocab_size, embed_size, thought_size, dropout,
                               bidirectional, pretrained_weight)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, sentences):
        thought_vectors = self.encoder(sentences)
        batch_size = len(thought_vectors)

        scores = torch.matmul(thought_vectors, torch.t(thought_vectors))
        scores[torch.eye(batch_size).byte()] = 0

        target_np = np.zeros(scores.size(), dtype=np.int64)
        for i in range(1, self.context_size + 1):
            # the i-th previous and next sentence
            target_np += np.eye(batch_size, k=-i, dtype=np.int64)
            target_np += np.eye(batch_size, k=i, dtype=np.int64)

        # normalize target matrix by row
        target_np_sum = np.sum(target_np, axis=1, keepdims=True)
        target_np = target_np / target_np_sum
        if self.device == torch.device('cpu'):
            target = torch.from_numpy(target_np).type(torch.LongTensor)
        else:
            target = torch.from_numpy(target_np).type(torch.cuda.LongTensor)

        loss = 0
        for t in target:
           loss += self.criterion(scores, t)

        return loss


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, thought_size, dropout,
                 bidirectional, pretrained_weight):
        super().__init__()

        self.thought_size = thought_size
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional

        if pretrained_weight:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weight,
                                                          freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(embed_size, thought_size, dropout=dropout,
                          bidirectional=bidirectional)

        self._init_weights(pretrained_weight)

    def _init_weights(self, pretrained_weight):
        if not pretrained_weight:
            self.embedding.weight.data.uniform_(-0.1, 0.1)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, sentences):
        word_embeddings = self.embedding(sentences).transpose(0, 1)

        _, hidden = self.gru(word_embeddings)

        if self.bidirectional:
            thought_vectors = torch.cat((hidden[0], hidden[1]), 1)
        else:
            thought_vectors = hidden[0]

        return thought_vectors
