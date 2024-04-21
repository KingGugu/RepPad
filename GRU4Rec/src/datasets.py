import torch
import random
from utils import neg_sample
from torch.utils.data import Dataset


class GRU4RecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.aug_type = args.aug_type

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(len(input_ids), dtype=torch.long),
        )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":

            if self.aug_type == 0:
                '''no repeated padding (original)'''
                input_ids = items[:-3]
                target_pos = items[1:-2]
                answer = [0]  # no use

            if self.aug_type == 1:

                '''random(1,max) repeated padding'''
                if len(items[:-3]) >= self.max_len:
                    input_ids = items[:-3]
                    target_pos = items[1:-2]
                else:
                    final_input_ids = items[:-3]
                    final_target_pos = items[1:-2]
                    max_num = int(self.max_len / len(items[:-3]))
                    pad_num = random.randint(1, max_num)
                    for i in range(pad_num):
                        final_input_ids = items[:-3] + final_input_ids
                        final_target_pos = items[1:-2] + final_target_pos
                    input_ids = final_input_ids
                    target_pos = final_target_pos
                answer = [0]  # no use

            if self.aug_type == 2:

                '''random(1,max) repeated padding with delimiter 0'''
                if len(items[:-3]) >= self.max_len:
                    input_ids = items[:-3]
                    target_pos = items[1:-2]
                else:
                    final_input_ids = items[:-3]
                    final_target_pos = items[1:-2]
                    max_num = int(self.max_len / len(items[:-3]))
                    pad_num = random.randint(1, max_num)
                    for i in range(pad_num):
                        final_input_ids = items[:-3] + [0] + final_input_ids
                        final_target_pos = items[1:-2] + [0] + final_target_pos
                    input_ids = final_input_ids
                    target_pos = final_target_pos
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
