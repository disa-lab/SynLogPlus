# Log Parsing with Generalization Ability under New Log Types
——Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 203)

Original code: https://github.com/gaiusyu/Log3T

## How to run

Use the `train.py` script to train the model on the log data.

```
python train.py         # For Loghub-2k dataset
python train.py -full   # For Loghub-2.0 dataset
```

Use the `evaluate.py` script to run the trained model on the log data.

```
python evaluate.py                      # Original Log3T (two-phase) for Loghub-2k dataset
python evaluate.py -full                # Original Log3T (two-phase) for Loghub-2.0 dataset
python evaluate.py --eval-training      # To enable only training (single-phase)
python evaluate.py --group-first        # To enable grouping before template identification (Log3T')
```

The results will be stored in the directories 'Result-two-phase', 'Result-group-first',
'Result-single-phase'.
