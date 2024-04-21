# -*- coding: utf-8 -*-
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_


class GRU4Rec(nn.Module):
    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        self.args = args

        # load dataset info
        self.max_seq_length = args.max_seq_length
        self.n_items = args.item_size

        # load parameters info
        self.embedding_size = args.hidden_size
        self.hidden_size = 128
        self.num_layers = 1
        self.dropout_prob = args.dropout_prob

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
