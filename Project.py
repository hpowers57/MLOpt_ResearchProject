import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim


np.random.seed(1)


ticker = 'AAPL'
data = yf.download(ticker, '2000-01-01', '2020-01-01') #gathering data from yahoo stocks webpage

# csv = ticker + ' CLosing Price.csv'
# data.to_csv(csv, index=False)

close_price = data['Close'] #creating the target variable for prediction


# plt.figure(figsize=(16, 6)) #graph to view the closing price over the entire time frame of the data

# plt.plot(close_price, color='blue')

# plt.xlabel('Date', fontsize=16)
# plt.ylabel('Closing Price', fontsize=16)

# title = 'Closing Price of ' + ticker
# plt.title(title, fontsize=20)

# plt.grid(True)

# plt.show()


n_timesteps = 200 #amount of days each bin will contain that the model will use to predict the next closing price

features_set = [] #creation of bins from original data
labels = []
for i in range(n_timesteps, close_price.shape[0]):
    features_set.append(close_price[i-n_timesteps:i])
    labels.append(close_price[i])
    

features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))


n = len(close_price) #finding the 70/30 split in the data for training and testing sets
n_train = int(n * 0.7)

features_set_train, features_set_test =  features_set[0:n_train,:,:], features_set[n_train:n,:,:]
labels_train, labels_test = labels[0:n_train], labels[n_train:n]


X_train = torch.from_numpy(features_set_train).type(torch.Tensor)
X_test = torch.from_numpy(features_set_test).type(torch.Tensor)

y_train = torch.from_numpy(labels_train).type(torch.Tensor)
y_test = torch.from_numpy(labels_test).type(torch.Tensor)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size[0], num_layers, batch_first=True)
        self.y1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.dropout1 = nn.Dropout(0.3)
        self.y2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.elu = nn.ELU(alpha=0.5)
        self.dropout2 = nn.Dropout(p=0.6)
        self.y3 = nn.Linear(hidden_size[2], output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size[0])
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size[0])
        
        hiddenStates, _ = self.lstm(x, (h0, c0))
        
        lstm_out = self.y1(hiddenStates[:, -1, :])
        lstm_out = self.dropout1(lstm_out)

        dense_out = self.y2(lstm_out)
        dense_out = self.elu(dense_out)
        dense_out = self.dropout2(dense_out)

        return self.y3(dense_out)
    

input_size = 1
hidden_size = [8, 70, 20]
output_size = 1
lr = 0.075
nepochs = 20

model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_history = np.zeros(nepochs)
for e in range(nepochs):
    for i in range(0, len(X_train), 1000):
        optimizer.zero_grad()

        y_train_pred = model(X_train)

        loss = criterion(y_train_pred, y_train)

        loss.backward()

        optimizer.step()

    loss_history[e] = loss.item()
    print("Epoch ", e+1, "MSE: ", loss.item())


plt.figure(figsize=(12, 6))

plt.plot(range(1, nepochs+1), loss_history, label='train MSE', color="blue")

plt.title('LSTM: min(train MSE) = {}'.format(str(round(min(loss_history), 6))), fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.legend(loc='upper right', fontsize=16)

plt.show()


# xhat = model(features_set) #predicting the next closing price for each bin in the original data

# window = np.arange(len(close_price)-len(labels_test), len(close_price)) #creating the window that the validation set is contained in

# plt.figure(figsize=(12, 6)) #graphing the validation closing prices along with the predicted closing price to compare how well the model did with predicting

# plt.plot(close_price.index[window], close_price[window], linestyle=':', marker='o', color='blue', label = "$x_t$")
# plt.plot(close_price.index[window], xhat[window-n_timesteps], linestyle=':', marker='o', color='red', label = "$\hat{x}_t$")

# plt.title("1 step ahead predictions by LSTM", fontsize=16)
# plt.legend(loc='lower left', fontsize=16)
# plt.grid(True)

# plt.show()


