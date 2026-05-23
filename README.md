# Multi-Head Attention integrated LCG (LSTM-CNN-GRU) model for Load Forecasting

The proposed architecture consists of two major branches:

1. CNN Branch: The CNN branch extracts local temporal patterns using -
Conv1D layers
MaxPooling layers
Dropout regularization
This branch captures hidden spatial-temporal representations from the input sequences.

2. LSTM-GRU Branch: The recurrent branch combines -
LSTM layers for long-term dependency learning
GRU layers for efficient sequential modeling
This branch captures both short-term and long-term temporal relationships.

3. Multi-Head Attention Mechanism: A custom Multi-Head Attention layer is implemented to improve feature learning.
The attention mechanism-
Focuses on important temporal information
Learns weighted feature representations
Enhances prediction capability from both CNN and recurrent branches
Outputs from the CNN and LSTM-GRU branches are passed through attention layers and concatenated before final prediction.


