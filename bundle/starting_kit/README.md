# NAS Unseen-Data 2026 Starting Kit
Hi, thanks for participating in the NAS Unseen-Data Competition!

To find out more information, including dates and rules, please visit our website: [https://www.nascompetition.com](https://www.nascompetition.com).

# Contents
The starting kit contains the following:
* `submission_template/`: This contains everything you need to implement to create a valid submission. See the included README within for more details
* `submission_example/`: Here's an example submission we made, for reference
* `using_extra_packages/`: Tutorials on how to include extra Python packages (e.g. gromo) in your submission
* `Makefile`: Some scripts that will let you build and test your submission locally against the same ingestion and scoring programs used on our servers, more details on this in the "Testing your Submission" section

# Datasets
The datasets used for the real Phase 2/3 evaluation are kept hidden until the end of the competition. However, below we have provided links to public datasets created for previous iterations of the competition -- these are for Phase 1 practice only and will not appear in the final evaluation.

Our pipeline and DataLoaders are expecting each dataset to be contained in its own folder with six NumPy files for the training, validation, and testing data, split between images and labels. Furthermore, a `metadata` file is expected containing the input shape, codename, benchmark, and number of classes. See the datasets we created (linked below), for the appropriate structure.

- AddNIST: [https://doi.org/10.25405/data.ncl.24574354.v1](https://doi.org/10.25405/data.ncl.24574354.v1)
- Language: [https://doi.org/10.25405/data.ncl.24574729.v1](https://doi.org/10.25405/data.ncl.24574729.v1)
- MultNIST: [https://doi.org/10.25405/data.ncl.24574678.v1](https://doi.org/10.25405/data.ncl.24574678.v1)
- CIFARTile: [https://doi.org/10.25405/data.ncl.24551539.v1](https://doi.org/10.25405/data.ncl.24551539.v1)
- Gutenberg: [https://doi.org/10.25405/data.ncl.24574753.v1](https://doi.org/10.25405/data.ncl.24574753.v1)
- GeoClassing: [https://doi.org/10.25405/data.ncl.24050256.v3](https://doi.org/10.25405/data.ncl.24050256.v3)
- Chesseract: [https://doi.org/10.25405/data.ncl.24118743.v2](https://doi.org/10.25405/data.ncl.24118743.v2)
- Sudoku: [https://doi.org/10.25405/data.ncl.26976121.v1](https://doi.org/10.25405/data.ncl.26976121.v1)
- Voxel: [https://doi.org/10.25405/data.ncl.26970223.v1](https://doi.org/10.25405/data.ncl.26970223.v1)
- Myofibre: [https://doi.org/10.25405/data.ncl.26969998.v1](https://doi.org/10.25405/data.ncl.26969998.v1)
- GameOfLife: [https://doi.org/10.25405/data.ncl.30000835](https://doi.org/10.25405/data.ncl.30000835)
- Cryptic: [https://doi.org/10.7488/ds/8054](https://doi.org/10.7488/ds/8054)
- Windspeed: [https://doi.org/10.7488/ds/8053](https://doi.org/10.7488/ds/8053)

# Writing Your Submission
In this competition, you will be asked to produce three components:
1. A DataProcessor, that takes in raw numpy arrays comprising the train/valid/test splits of the dataset and creates train/valid/test PyTorch dataloaders. These can perform whatever preprocessing or augmentation that you might want
2. A NAS algorithm, that takes in the dataloaders and produces some optimal PyTorch model
3. A Trainer, that trains that optimal model over the train dataloader

In general, the following pipeline occurs for each dataset:
1. Raw Dataset -> `DataProcessor` -> Train, Valid, and Test dataloaders
2. Train Dataloader + Valid Dataloaders -> `NAS` -> Model
3. Model + Train Dataloader + Valid Dataloaders -> `Trainer.train` -> Fully-trained model
4. Fully-trained model + Test Dataloader -> `Trainer.predict` -> Predictions

See `submission_template/README.md` for specifics about how to write these, and `submission_example/` for an example valid submission.

# Runtime
Our ingestion program (`ingestion_program/ingestion.py` on our servers) creates a `clock` object that is passed into the `__init__` of your `DataProcessor`, `NAS`, and `Trainer` classes, and a `time_remaining` field is added to the dataset metadata before each stage. Use these to check the compute time remaining and adapt your pipeline accordingly.

The total compute budget is shared across all datasets in a run: **1 hour in Phase 2** (this is a smoke test, just to confirm your submission runs end-to-end without crashing) and a longer, undisclosed budget in **Phase 3** (the real, scored run). Your last working Phase 2 submission is automatically reused for Phase 3 -- there is no separate Phase 3 submission. It is your job to use the clock to manage the amount of time your code has and to adapt to the amount of time given.

# Testing Your Submission
The included Makefile will let you test your submission via the same ingestion and scoring programs our servers use. If the Makefile works, then you can be fairly confident your submission will work on our machines. However, you should still be
careful about things like package imports, because trying to import something that doesn't exist in our environment will break your submission.

To test your submission from start-to-finish, run:

`make submission=$SUBMISSION_DIRECTORY all`


For example, to run the example submission:

`make submission=submission_example all`


# Submitting
To bundle your submission, run:

`make submission=$SUBMISSION_DIRECTORY zip`

Then submit the zip file through the "My Submissions" tab on the Codabench competition page. If you have any questions, reach out to us at [nas-competition-contact@newcastle.ac.uk](mailto:nas-competition-contact@newcastle.ac.uk).
