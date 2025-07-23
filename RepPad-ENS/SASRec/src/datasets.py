# -*- coding: utf-8 -*-

import torch
import random
from utils import neg_sample
from torch.utils.data import Dataset


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.aug_type = args.aug_type

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified

        target_neg_list = []
        for i in range(self.args.neg_seq_num):
            target_neg = []
            seq_set = set(items)
            for _ in input_ids:
                target_neg.append(neg_sample(seq_set, self.args.item_size))
            pad_len = self.max_len - len(input_ids)
            target_neg = [0] * pad_len + target_neg
            target_neg = target_neg[-self.max_len:]
            assert len(target_neg) == self.max_len
            target_neg_list.append(target_neg)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg_list, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        if self.data_type == "train":

            if self.aug_type == 0:
                '''no repeat pad (original)'''
                input_ids = items[:-3]
                target_pos = items[1:-2]
                answer = [0]  # no use

            if self.aug_type == 1:

                '''random repeat pad'''
                if len(items[:-3]) >= self.max_len:
                    input_ids = items[:-3]
                    target_pos = items[1:-2]
                else:
                    temp_input_ids = items[:-3]
                    temp_target_pos = items[1:-2]
                    max_num = int(self.max_len / len(items[:-3]))
                    pad_num = random.randint(1, max_num)
                    input_ids = temp_input_ids * pad_num + temp_input_ids
                    target_pos = temp_target_pos * pad_num + temp_target_pos
                answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
