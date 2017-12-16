# DeepProtease
I know it's kind of a mess, but this is more of just an exploratory project to learn the basics of TensorFlow and apply it to a real-world problem. I've learned a few lessons, and my next TensorFlow project should be much cleaner. I may end up refactoring later if I have time.
## The problem
HIV-1 protease cleaves new polyproteins into smaller functional proteins as part of its life cycle. The dataset is about 6000 amino acid sequences, each 8 amino acids long, labeled with whether the protease enzyme is able to cleave that target. The neural network is trained on a subset of this data, and tested on a subset of the data to make predictions.

To train and run the model, run the command:
```
$ ./sparse_encoding_model.py
```