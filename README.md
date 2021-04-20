# feature-selection
This repository contains implementation of permutation importance based feature selection.
To run the pipeline for training an evaluation of the framework run:
```shell
$ python3 main.py --test_no=1
```
It needs a file located in ./LIME/data/constituents.csv
with the stock set it will build the feature selection on.

### Outputs
A metric file found in ./LIME/data/LOOC_metrics_cr_{test_no}.csv
that has the following format:

MSE_baseline | MAE_baseline | walk | ticker | MSE_pi | MAE_pi | model | selection_error | removed_FI | removed_error | removed_column 
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- 
0.00019|0.01085 |1.0 | FP.PA | 0.00027 | 0.01210 | pi|  | -1.8409e-05 | 0.00827| Returns_3

with one line for each tuple (walk,ticker,data_type)
### Example command

```shell
$ python3 main.py 
--data_type pi --start_date 2007-01-01 --end_date 2017-01-01 --no_walks 7
--no_features 1 --train_length 4 --test_length 1
--num_rounds 50 --test_no 11
```

