## RepPad

Official source code for RecSys 2024 paper [Repeated Padding for Sequential Recommendation](https://arxiv.org/abs/2403.06372v2) (RepPad) and [Repeated Padding+: Simple yet Effective Data Augmentation Plugin for Sequential Recommendation](https://arxiv.org/abs/2403.06372v3) (RepPad+).

This repository provides implementations for two representative sequential recommendation backbones: `GRU4Rec` and `SASRec`. RepPad and RepPad+ are implemented as plug-and-play sequence augmentation strategies controlled by `--aug_type`.

## Supported Datasets

Current built-in datasets under both `GRU4Rec/data` and `SASRec/data`:

- `Beauty`
- `Home`
- `Sports_and_Outdoors`
- `Toys_and_Games`
- `Yelp`
- `LastFM`
- `ml-1m`

## Run the Original Model and RepPad

Go to the `src` folder in either `SASRec` or `GRU4Rec`, then run:

- `--aug_type=0`: original model with traditional zero padding
- `--aug_type=1`: RepPad with random(`1`, `max`) repeated padding
- `--aug_type=2`: RepPad with random(`1`, `max`) repeated padding and delimiter `0`

General command template:

```bash
python main.py --data_name=<DATASET> --aug_type=<AUG_TYPE> --model_idx=<RUN_ID>
```

RepPad commands (`aug_type=0/1/2`) on all current datasets:

```bash
python main.py --data_name=Beauty --aug_type=0 --model_idx=3
python main.py --data_name=Home --aug_type=0 --model_idx=3
python main.py --data_name=Sports_and_Outdoors --aug_type=0 --model_idx=3
python main.py --data_name=Toys_and_Games --aug_type=0 --model_idx=3
python main.py --data_name=Yelp --aug_type=0 --model_idx=3
python main.py --data_name=LastFM --aug_type=0 --model_idx=3
python main.py --data_name=ml-1m --aug_type=0 --model_idx=3

python main.py --data_name=Beauty --aug_type=1 --model_idx=4
python main.py --data_name=Home --aug_type=1 --model_idx=4
python main.py --data_name=Sports_and_Outdoors --aug_type=1 --model_idx=4
python main.py --data_name=Toys_and_Games --aug_type=1 --model_idx=4
python main.py --data_name=Yelp --aug_type=1 --model_idx=4
python main.py --data_name=LastFM --aug_type=1 --model_idx=4
python main.py --data_name=ml-1m --aug_type=1 --model_idx=4

python main.py --data_name=Beauty --aug_type=2 --model_idx=5
python main.py --data_name=Home --aug_type=2 --model_idx=5
python main.py --data_name=Sports_and_Outdoors --aug_type=2 --model_idx=5
python main.py --data_name=Toys_and_Games --aug_type=2 --model_idx=5
python main.py --data_name=Yelp --aug_type=2 --model_idx=5
python main.py --data_name=LastFM --aug_type=2 --model_idx=5
python main.py --data_name=ml-1m --aug_type=2 --model_idx=5
```

## Run the RepPad+

Go to the `src` folder in either `SASRec` or `GRU4Rec`, then run:

- `--aug_type=3`: RepPad+ with random(`1`, `max`) repeated padding plus medium-length subsequence padding (with delimiter `0`)
- `--aug_type=4`: RepPad+ with random(`1`, `max`) repeated padding plus medium-length subsequence padding (without delimiter `0`)

RepPad+ commands (`aug_type=3/4`) on all current datasets:

```bash
python main.py --data_name=Beauty --aug_type=3 --model_idx=6
python main.py --data_name=Home --aug_type=3 --model_idx=6
python main.py --data_name=Sports_and_Outdoors --aug_type=3 --model_idx=6
python main.py --data_name=Toys_and_Games --aug_type=3 --model_idx=6
python main.py --data_name=Yelp --aug_type=3 --model_idx=6
python main.py --data_name=LastFM --aug_type=3 --model_idx=6
python main.py --data_name=ml-1m --aug_type=3 --model_idx=6

python main.py --data_name=Beauty --aug_type=4 --model_idx=7
python main.py --data_name=Home --aug_type=4 --model_idx=7
python main.py --data_name=Sports_and_Outdoors --aug_type=4 --model_idx=7
python main.py --data_name=Toys_and_Games --aug_type=4 --model_idx=7
python main.py --data_name=Yelp --aug_type=4 --model_idx=7
python main.py --data_name=LastFM --aug_type=4 --model_idx=7
python main.py --data_name=ml-1m --aug_type=4 --model_idx=7
```

## Hyperparameters

Table 4 from `revise/main.pdf` (Hyper-parameter settings and tuning ranges of main backbone models):

| Hyper-parameter | GRU4Rec | NARM | Caser | NextItNet |
|---|---|---|---|---|
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| embedding_size | 64 | 64 | 64 | 64 |
| early_stopping | 20 | 20 | 20 | 20 |
| num_layers | {1, 2, 3} | 1 | - | {4, 5, 6} |
| hidden_size | 128 | 128 | - | - |
| dropout_prob | [0.1, 0.5] | 0.25 | [0.1, 0.5] | - |
| reg_weight | - | - | {1e-1, 1e-2, 1e-3, 1e-4} | {1e-3, 1e-4, 1e-5, 0} |
| concatenation_dropout | - | 0.5 | - | - |
| vertical_filters | - | - | {4, 8, 16, 32, 64} | - |
| horizontal_filters | - | - | {1, 2, 4, 8, 16} | - |
| kernel_size | - | - | - | 3 |
| dilations | - | - | - | 1->4 |

| Hyper-parameter | SASRec | LightSANs | gMLP | FMLP-Rec |
|---|---|---|---|---|
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| embedding_size | 64 | 64 | 64 | 64 |
| early_stopping | 20 | 20 | 20 | 20 |
| num_layers | {2, 3, 4} | {2, 3, 4} | {2, 3, 4} | {2, 3, 4} |
| num_heads | {2, 3, 4} | {2, 3, 4} | - | - |
| inner_size | 256 | 256 | 256 | 256 |
| hidden_dropout_prob | [0.1, 0.5] | [0.1, 0.5] | [0.1, 0.5] | [0.1, 0.5] |
| attn_dropout_prob | [0.1, 0.5] | [0.1, 0.5] | - | - |
| hidden_act | gelu | gelu | gelu | gelu |
| layer_norm_eps | 1e-12 | 1e-12 | 1e-12 | 1e-12 |
| initializer_range | 0.02 | 0.02 | 0.02 | 0.02 |
| k_interests | - | 5 | - | - |

For RepPad/RepPad+, the backbone hyper-parameters are kept the same as their corresponding original models for fair comparison. Code implementations of other backbones will be released after the paper is accepted.

## Pseudocode

The following pseudocode is adapted from `revise/main.pdf` and aligned with this repository implementation.

RepPad (`aug_type=1/2`, random(`1`, `max`)):

```python
def reppad(orig_seq, max_len, use_delimiter=True):
    final_input = orig_seq[:-3]      # training items
    final_target = orig_seq[1:-2]    # target items

    if int(max_len / len(final_input)) <= 1:
        # insufficient remaining space for repeated full-sequence padding
        return final_input, final_target

    max_pad_num = int(max_len / len(final_input))
    pad_num = random.randint(1, max_pad_num)

    if use_delimiter:
        final_input = (final_input + [0]) * pad_num + final_input
        final_target = (final_target + [0]) * pad_num + final_target
    else:
        final_input = final_input * pad_num + final_input
        final_target = final_target * pad_num + final_target

    return final_input, final_target
```

RepPad+ (`aug_type=3/4`, random(`1`, `max`), with medium-length handling):

```python
def reppad_plus(orig_seq, max_len, use_delimiter=True):
    final_input = orig_seq[:-3]      # training items
    final_target = orig_seq[1:-2]    # target items

    if use_delimiter:
        no_space_cond = len(final_input) > max_len - 2
        sub_len = max_len - 1 - len(final_input)
    else:
        no_space_cond = len(final_input) > max_len - 1
        sub_len = max_len - len(final_input)

    if no_space_cond:
        # no room for RepPad+ operation
        return final_input, final_target

    if int(max_len / len(final_input)) <= 1:
        # medium-length case: pad truncated consecutive subsequence
        start_index = random.randint(0, len(final_input) - sub_len)
        sub_input = final_input[start_index:start_index + sub_len]
        sub_target = final_target[start_index:start_index + sub_len]
        if use_delimiter:
            return sub_input + [0] + final_input, sub_target + [0] + final_target
        return sub_input + final_input, sub_target + final_target

    # short-length case: regular repeated padding
    max_pad_num = int(max_len / len(final_input))
    pad_num = random.randint(1, max_pad_num)
    if use_delimiter:
        return (final_input + [0]) * pad_num + final_input, (final_target + [0]) * pad_num + final_target
    return final_input * pad_num + final_input, final_target * pad_num + final_target
```

In practice, after applying RepPad/RepPad+, you should still run the final sequence preparation step (truncate/pad to `max_len`) before feeding sequences into the model.

## Log Files

We provide log files and trained weights for part of the experiments in `src/output` (currently focused on the five short-sequence datasets used in the original RepPad setting). In our naming convention, `xxxxx-1.txt` is the original model and `xxxxx-2.txt` is the model with RepPad.

## Acknowledgement

- Training pipeline is implemented based on [CoSeRec](https://github.com/YChen1993/CoSeRec).
- SASRec model are implemented based on [RecBole](https://github.com/RUCAIBox/RecBole).

Thanks for their efficient open-source implementation.

## Reference

Please cite our paper if you use this code.

```
@inproceedings{dang2024repeated,
  title={Repeated Padding for Sequential Recommendation},
  author={Dang, Yizhou and Liu, Yuting and Yang, Enneng and Guo, Guibing and Jiang, Linying and Wang, Xingwei and Zhao, Jianzhe},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={497--506},
  year={2024}
}
```
