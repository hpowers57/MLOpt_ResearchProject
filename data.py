import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

def pull_data(ticker, start="2000-01-01", end="2020-01-01", scale=True):
  data = yf.download(ticker, "2000-01-01", "2020-01-01")
  price = data["Close"]
  
  if scale:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price = scaler.fit_transform(price.values.reshape(-1, 1))

  return price

def make_train_test(data, frac=0.7, bin_size=14):
  seqs = list()
  for idx in range(data.shape[0] - bin_size):
      seqs.append(price[idx:idx+bin_size])
      
  seqs = np.array(seqs)
  train_n = int(np.floor(frac * seqs.shape[0]))
  test_n = seqs.shape[0] - train_n

  train_x = torch.from_numpy(seqs[:train_n,:-1,:]).type(torch.Tensor)
  train_y = torch.from_numpy(seqs[:train_n,-1,:]).type(torch.Tensor)
  test_x = torch.from_numpy(seqs[train_n:,:-1,:]).type(torch.Tensor)
  test_y = torch.from_numpy(seqs[train_n:,-1,:]).type(torch.Tensor)
  
  return train_x, train_y, test_x, test_y
