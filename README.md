## Anonymous code for Review

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

We also provide the log files and trained weights on these five datasets in the `src/output` directory. 
`xxxxx-1.txt` is the performance of the original model, `xxxxx-2.txt` is the performance after adding RepPad.
