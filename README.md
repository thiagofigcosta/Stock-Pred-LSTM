# Stock-Pred-LSTM

## To run

>- To download referecence datasets
```
python3 Pytho\{N\}.py stock_pred.py --download-datasets
```

>- To train the test models
```
python3 Pytho\{N\}.py stock_pred.py --train-all-test-models --dataset-paths datasets/GE_I1d_F0_T2020-10.csv
```

>- To restore best trained checkpoint
```
python3 Pytho\{N\}.py stock_pred.py --restore-best-checkpoints
```

>- To run and plot the trained test models
```
python3 Pytho\{N\}.py stock_pred.py --test-all-test-trained-models --dataset-paths datasets/GE_I1d_F0_T2020-10.csv
```

>- To download some dataset
```
python3 Pytho\{N\}.py stock_pred.py --download-stock --use-hour-interval --stock-name GE --start-date 23/02/2019 --end-date 15/07/2019
```

>- To run research questions, use `--train_without_plot` for training, and then run without it to see results
```
python3 Pytho\{N\}.py stock_pred.py --qp1
python3 Pytho\{N\}.py stock_pred.py --qp2
python3 Pytho\{N\}.py stock_pred.py --qp3
python3 Pytho\{N\}.py stock_pred.py --qp4
python3 Pytho\{N\}.py stock_pred.py --qp6
```

## Wildcard to isolate qp models
>- QP1
```
mkdir qp1
cp saved_models/*_I1d_F0_T2020-10* qp1/
rm -rf qp1/*IFs6*
rm -rf qp1/*trunc*
rm -rf qp1/*+*
```

>- QP2
```
mkdir qp2
cp saved_models/*IFs6* qp2/
```

>- QP3
```
mkdir qp3
cp saved_models/*trunc* qp3/
cp saved_models/*+* qp3/
```

>- QP4
```
mkdir qp4
cp saved_models/*I1h* qp4/
```