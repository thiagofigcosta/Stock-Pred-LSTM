# Stock-Pred-LSTM

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