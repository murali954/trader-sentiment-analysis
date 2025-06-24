# Trader Sentiment Analysis

This project explores the relationship between trader performance and market sentiment using two datasets: trader execution data and Bitcoin fear/greed sentiment.

## Structure
- `data/`: Raw data files
- `notebooks/`: Jupyter notebooks for analysis
- `outputs/`: Generated plots and results

## How to Run
```bash
pip install -r requirements.txt
python main.py

```
Data Quality Check:
Total records after merge: 211224
Records with sentiment data: 211218
Records missing sentiment: 6
Records after removing missing sentiment: 211218

Sentiment distribution:
classification
Fear             61837
Greed            50303
Extreme Greed    39992
Neutral          37686
Extreme Fear     21400
Name: count, dtype: int64

Basic statistics:
       Execution Price  ...     Closed PnL
count    211218.000000  ...  211218.000000   
mean      11415.047529  ...      48.549304   
std       29448.010305  ...     917.989791   
min           0.000005  ... -117990.104100   
25%           4.858550  ...       0.000000   
50%          18.280000  ...       0.000000   
75%         101.895000  ...       5.790132   
max      10

![image](https://github.com/user-attachments/assets/167f062f-ebf0-4298-b4f3-619916ac3e70)
