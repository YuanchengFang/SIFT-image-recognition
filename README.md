Scene Recognition with Bag of Words
==================

Run codes
----
Install packages:
```shell
pip install requirements.txt
```

Run with tiny images method
```shell
python main.py --method tiny
```

Run with SIFT bag of words method
```shell
python main.py --method sift
```

Analyze the relation between vocabulary size and performance

```shell
python main.py --analyze vocab
```

Analyze performance on different categories (SIFT)

```shell
python main.py --analyze category
```

Analyze performance on different categories (Tiny Images)

```shell
python main.py --analyze tiny
```



`plot.ipynb` is used to plot images I use in my report.

Note
----

- Input train data and test data are in `data.zip`
- May need to change the `TRAIN_DIR` and `TEST_DIR` in `main.py`, to specify input directory.