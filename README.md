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

Note
----
- Train data and test data are in `data.zip`
- You may need to change the `TRAIN_DIR` and `TEST_DIR` in `main.py`, to specify input directory.