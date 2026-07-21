# Instructions

### Technical Details

A great place to start is to look at the template and example submissions, to get an idea of what each of three files we're asking you to write should look like. Additionally, the evaluation directory in the Starting Kit contains the evaluation scripts that our machines will be using, so you can see exactly how it all works. However, the following section will also quickly go through everything that you will need to do.

### How a submission works

As a starting point, check out the template submission within the Starting Kit. To transform the template into a valid submission, there are a number of functions that need to be implemented. Check the individual files within the template to see exactly how it all works, and there is more documentation within the files that explain everything in more depth.

For a valid submission, you are asked to implement the following functions within the following classes:

- **`DataProcessor`:**
  - `__init__()`: This function receives raw data in the form of numpy arrays for the train, valid, and test data, as well the dataset metadata
  - `process()`: This function must output 3 PyTorch dataloaders for the train, valid, and test data splits
- **`NAS`:**
  - `__init__()`: This function receives the dataloaders created by the `DataProcessor`, and the dataset metadata
  - `search()`: This function should search for an optimal architecture for this dataset, and should output a PyTorch model
- **`Trainer`:**
  - `__init__()`: This function receives the dataloaders created by the `DataProcessor`, and the model produced by the `NAS` class
  - `train()`: This function should fully train your model and return it
  - `predict()`: This function should produce a list of predicted class labels over the `test_dataloader`

**Do not include the evaluation files `main.py` or `score.py` in your submission or include any files named `main.py` or `score.py`, any such files will be overwritten and may invalidate your submission**

### Evaluation Pipeline

In general, the evaluation script runs the following pipeline for each dataset:

1. The Raw Dataset is passed to the `DataProcessor` and produces Train, Valid, and Test dataloaders
2. The train and valid dataloaders are passed to `NAS`, which outputs a model
3. The model, the train and valid dataloaders are passed to the `Trainer.train` function, which outputs a fully trained model
4. The fully-trained model and test loader and passed to the `Trainer.predict` function, which outputs the class predictions for each image in the test loader

### Tips and Tricks

#### Datasets

Each of three datasets in the competition will be an n-class classification task over 4-D images of shape `(#Images, Channels, Height, Width)`. Each dataset has a pre-divided splits for training, validation, and testing, each of which are labeled accordingly.

Additionally, each dataset will be accompanied by a metadata dictionary, that contains the following information:

- **`num_classes`**: The total number of classes in the classification problem
- **`input_shape`**: The shape of the `train_x` data. All images in each split will have the same channel count, heigh, and width, but the different splits will have different numbers of images
- **`codename`**: A unique codename for this dataset to refer to it throughout the competition
- **`benchmark`**: The benchmark classification accuracy for this dataset. This is the score that our example submission achieved on the dataset, and is the mark necessary to score 0 points on this dataset. Accuracies above the benchmark will score more points, up to a total of **10 points** for a perfect 100% test accuracy. Conversely, accuracies below the benchmark will score negative points, up to **-10** at worst

#### Designing your pipeline

Each of three pipeline classes (`DataProcessor`, `NAS`, and `Trainer`) will receive the dataset metadata dictionary in their initialization. You can alter this however you want, in case you want to pass messages between your various classes.

Make sure to evaluate your pipeline over a variety of datasets, to ensure that it is flexible enough to work well on a variety of tasks. Make sure not to specifically tailor your pipeline to the datasets bundled with the Starting Kit, because none of them will appear in the final evaluation round. The three datasets that we will use to evaluate your submission have been designed from scratch for this competition and will be kept secret until after the competition.

We believe NAS pipelines should be responsive to the memory of the machine they are being run on. Following this belief, we have decided to withold the GPU that we will use to evaluate final submission (though each submission will be tested on machines of equivelent specs). We encourage you to make your submission adaptable to the amount of memory available, especially as the datasets may (and past datasets have) wildly different file sizes.

### Submission Runtime limit

**Your submission will have 24 hours total to run on Codabench servers.** That means it needs to perform the entire NAS pipeline, training, and test prediction for each of the three final datasets within 24 hours. **If your submission exceeds this time, it will be instantly terminated and will receive no score.** To help you keep aware of this, the evaluation pipeline will add a field to the metadata dictionary called `time_remaining`. This is an estimate of the remaining time your submission has in seconds. You can use this to early-stop your algorithm, tailor your training epochs, adjust your search algorithm, whatever you need to do to ensure your submission runs in under 24 hours.

### Other

If you run into any problems or simply have questions, feel free to reach out to us! You can email us at: [nas-competition-contact@newcastle.ac.uk](mailto:nas-competition-contact@newcastle.ac.uk).


## Partners & Sponsors

<div align="center">

<img src="https://www.nascompetition.com/images/NAIL.svg" height="80" style="margin: 10px 15px;">
<img src="https://www.nascompetition.com/images/automl_highres.png" height="80" style="margin: 10px 15px;">

<br><br>

<img src="https://www.nascompetition.com/images/newc-trimed.png" height="80" style="margin: 10px 15px;">
<img src="https://www.nascompetition.com/images/Durham_University_Logo.png" height="80" style="margin: 10px 15px;">
<img src="https://www.nascompetition.com/images/edin-trimed.png" height="80" style="margin: 10px 15px;">
<br><br>
<img src="https://www.nascompetition.com/images/UoG_colour.png" height="80" style="margin: 10px 15px;">
<img src="https://www.nascompetition.com/images/hzb-logo.svg" height="80" style="margin: 10px 15px;">

</div>