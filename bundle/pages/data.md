# Data

This is an AutoML challenge: your NAS pipeline is submitted once and evaluated, unmodified, across multiple unseen image-classification datasets.

- **Phase 1:** develop and test your pipeline locally using the public practice datasets described below.
- **Phase 2:** your submission runs on our servers against secret, unseen datasets, as a short smoke-test.
- **Phase 3:** your last working Phase 2 submission is rerun against the same secret datasets for the final, scored evaluation.

The three datasets described below are from previous years' challenges, provided for Phase 1 practice only -- they will not appear in the Phase 2/3 evaluation.

## Mateo (MultNIST)

The MultNIST dataset is a constructed dataset from MNIST Images. The intention of this dataset is to require machine learning models to do more than just image classification but also perform a calculation, in this case multiplication followed by a mod operation. For each image, three MNIST Images were randomly chosen and combined together through the colour channels, resulting in a three colour-channel image so each MNIST image represents one colour channel.

The data is in a channels-first format with a shape of `(n, 3, 28, 28)` where `n` is the number of samples in the corresponding set (50,000 for training, 10,000 for validation, and 10,000 for testing).

There are ten classes in the dataset, with 7,000 examples of each, distributed evenly between the three subsets.

The label of each image is generated using the formula `(r * g * b) % 10` where r, g, and b are the red, green, and blue colour channels respectively. An example of a MultNIST image would be an rgb configuration of 3, 7, and 4 respectively, which would result in a label of 4 (`(3 * 7 * 4) % 10`).

## LaMelo (Language)

The Language dataset is a constructed dataset using words from aspell dictionaries. The intention of this dataset is to require machine learning models to not only perform image classification but also linguistic analysis to figure out which letter frequency is associated with each language. For each Language image, four six-letter words using the standard Latin alphabet were selected, with any words containing diacritics (such as é or ü) or the letters 'y'/'z' removed.

These words are encoded on a graph with one axis representing the index of the 24-character-long string (the four words joined together) and the other representing the letter (going A-X).

The data is in a channels-first format with a shape of `(n, 1, 24, 24)` where `n` is the number of samples in the corresponding set (50,000 for training, 10,000 for validation, and 10,000 for testing).

There are ten classes in the dataset, with 7,000 examples of each, distributed evenly between the three subsets, one per language:

English: 0, Dutch: 1, German: 2, Spanish: 3, French: 4, Portuguese: 5, Swahili: 6, Zulu: 7, Finnish: 8, Swedish: 9

## Adaline (AddNIST)

The AddNIST dataset is a constructed dataset from MNIST Images. The intention of this dataset is to require machine learning models to do more than just image classification but also perform a calculation, in this case addition. For each image, three MNIST Images were randomly chosen and combined together through the colour channels, resulting in a three colour-channel image so each MNIST image represents one colour channel.

The data is in a channels-first format with a shape of `(n, 3, 28, 28)` where `n` is the number of samples in the corresponding set (45,000 for training, 15,000 for validation, and 10,000 for testing).

There are twenty classes in the dataset, with 3,500 examples of each, distributed evenly between the three subsets.

The label of each image is generated using the formula `(r + g + b) - 1` where r, g, and b are the red, green, and blue colour channels respectively. An example of an AddNIST image would be an rgb configuration of 3, 7, and 4 respectively, which would result in a label of 13 (`(3 + 7 + 4) - 1`).
