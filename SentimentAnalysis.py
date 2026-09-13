import pandas as pd

SentimentDS = pd.read_csv("sentiment/AAPL.csv", header=0, index_col="Date", parse_dates=True)
S_mean = SentimentDS["ts_polarity"].mean()
print(S_mean)
SentimentDS["ts_polarity"] = SentimentDS["ts_polarity"].fillna(S_mean)
print(SentimentDS["ts_polarity"])

print(SentimentDS)