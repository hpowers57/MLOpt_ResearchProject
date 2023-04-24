# CSCI 6968 Research Project
## Models
All models are currently implemented in `models.py`. The current models include:
- Martingale (Elliot et al, 2017)
- RNN
- GRU
- LSTM
- ls-GRU

All RNN models are implemented with Torch.

## Data
Data is collected from Yahoo Finance with `pull_data` in `data.py`. Currently, we use the prices for Apple (AAPL) from January 1, 2000 to January 1, 2017 for training and January 1, 2017 to January 1, 2020 for validation. The data is split into bins and training and validation sets with `make_train_test`. We then test the models on Wells Fargo (WFC), Tesla (TSLA), General Mills (GIS) and American Tower Corp (AMT).

## Evaluation Metrics
Models are evaluated by their RMSE, R-squared, optimism ratio, and pessimism ratios. The implementation for the optimism and pessimism ratios can be found in `evaluation_metrics.py`. The optimism and pessimism ratios are defined by Sethia & Raut (2018).

## Experiments
The models above are trained and evaluated in `experiments.ipynb`.

## References
Aaron Elliot, Cheng Hsu, and Jennifer Slodoba. Time series prediction: Predicting stock price.
*arXiv*, 2017.

Akhil Sethia and Purva Raut. Application of lstm, gru and ica for stock price prediction. *Interna-
tional Conference on ICT for Intelligent Systems*, 2:479–487, 2018.
