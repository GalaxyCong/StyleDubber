"""
You may change these hyperparameters depending on the task.
"""
sma_head = 4
sma_dropout = 0.1
sma_tunable = False # If True, the stepwise monotonice multihead attention is activated. Else, it is a normal multihead attention just like in Transformer.