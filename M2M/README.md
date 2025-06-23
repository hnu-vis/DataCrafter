# Usage Instructions

## Using Demo Data (Pets Dataset)
If you are using the demo data (Pets dataset), the required data and files have already been saved in this project. You only need to run the following command to obtain the results:

```sh
$ python train.py
```

## Using Your Own Data
If you need to use your own data, you must run `/experiments/tree_cut.py` to generate intermediate data. The following steps must be completed:

### Fill in the Actual Information
You should provide your personal API key and the actual data path at lines 769 and 770 in the `tree_cut.py` file. For example:

```python
api_key = "your_api_key"
image_folder = "your_image_folder"
```

### Generate Intermediate Data
Run the following command to generate the intermediate data:

```sh
$ python /experiments/tree_cut.py
```

After completing these steps, you can find all the generated intermediate data files in the `backend/data` directory. Then, you can run the following command to obtain the results:

```sh
$ python train.py
```