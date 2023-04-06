# CSCI 6968 Research Project
## Models
All models are currently implemented in `models.py`. The current models include:
- RNN
- GRU

All models are implemented with Torch.

## Data
Either a folder with the pre-processed data or a file which collects and pre-processes the data. Include a brief description of the dataset(s) included.

## Evaluation Metrics
Models are evaluated by their RMSE, R-squared, optimism ratio, and pessimism ratios. THe implementation for the optimism and pessimism ratios can be found in `evaluation_metrics.py`. The optimism and pessimism ratios are defined by Sethia & Raut (2018).

## Experiments
File which trains and tests the model(s) for the datasets above and compiles the results tables.

## References
Akhil Sethia and Purva Raut. Application of lstm, gru and ica for stock price prediction. *Interna-
tional Conference on ICT for Intelligent Systems*, 2:479–487, 2018.
