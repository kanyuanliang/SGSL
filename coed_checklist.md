# Graph Structure Learning Based on Novel Regularization for Incremental Edge Pruning

This paper elaborates on how to utilize the code and data provided. The paper is available in two versions: one in .md format and the other in .pdf format, with both versions containing identical content.

## Requirements

The code provided in the supplementary materials runs on Python and requires the following packages: torch.

## Training

To train the GNN model, please read the comments from lines 312 to 338 in train.py, and run this command:

```train
python train.py
```

The trained model are saved in <font color="OrangeRed">'output/'</font>.

Uncomment lines 6-7 in the SGSL.py file to select the SGSL-enhanced GNN model.

