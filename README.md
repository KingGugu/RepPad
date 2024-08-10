# RepPad
Official source code for RecSys 2024 paper: [Repeated Padding for Sequential Recommendation](https://arxiv.org/abs/2403.06372)

We provide implementations of the two most representative sequential recommendation models, GRU4Rec and SASRec. You can quickly apply RepPad to your sequential model based on the pseudo-code provided in the paper.

# Run the Code

Go to the `src` folder in the `SASRec` or `GRU4Rec` directory, then run the following commands. 

`--aug_type=0` represents not using RepPad (traditional padding).
`--aug_type=1` represents using random(1,max) repeated padding.
`--aug_type=2` represents using random(1,max) repeated padding with delimiter 0.

```
python main.py --data_name=Toys_and_Games --aug_type=0 --model_idx=3
python main.py --data_name=Beauty --aug_type=0 --model_idx=3
python main.py --data_name=Sports_and_Outdoors --aug_type=0 --model_idx=3
python main.py --data_name=Home --aug_type=0 --model_idx=3
python main.py --data_name=Yelp --aug_type=0 --model_idx=3

python main.py --data_name=Toys_and_Games --aug_type=1 --model_idx=4
python main.py --data_name=Beauty --aug_type=1 --model_idx=4
python main.py --data_name=Sports_and_Outdoors --aug_type=1 --model_idx=4
python main.py --data_name=Home --aug_type=1 --model_idx=4
python main.py --data_name=Yelp --aug_type=1 --model_idx=4

python main.py --data_name=Toys_and_Games --aug_type=2 --model_idx=5
python main.py --data_name=Beauty --aug_type=2 --model_idx=5
python main.py --data_name=Sports_and_Outdoors --aug_type=2 --model_idx=5
python main.py --data_name=Home --aug_type=2 --model_idx=5
python main.py --data_name=Yelp --aug_type=2 --model_idx=5
```


# Log Files

We also provide some log files and trained weights on these five datasets of `SASRec` in the `src/output` directory. 
`xxxxx-1.txt` is the performance of the original model, `xxxxx-2.txt` is the performance after adding RepPad.


# Acknowledgement
 - Training pipeline is implemented based on [CoSeRec](https://github.com/YChen1993/CoSeRec).
 - SASRec model are implemented based on [RecBole](https://github.com/RUCAIBox/RecBole). 

Thanks them for providing efficient implementation.


# Reference

Please cite our paper if you use this code.
```
@article{dang2024repeated,
  title={Repeated Padding for Sequential Recommendation},
  author={Dang, Yizhou and Liu, Yuting and Yang, Enneng and Guo, Guibing and Jiang, Linying and Wang, Xingwei and Zhao, Jianzhe},
  journal={arXiv preprint arXiv:2403.06372},
  year={2024}
}
```
