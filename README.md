Online Feature Selection
=========================

The attached scripts provide two alternative utilities of the algorithm as described in article by Sengupta, D.; Bandyopadhyay, S.; Sinha, D., "A Scoring Scheme for Online Feature Selection: Simulating Model Performance Without Retraining," in Neural Networks and Learning Systems, IEEE Transactions on , vol.PP, no.99, pp.1-10
doi: 10.1109/TNNLS.2016.2514270

Kindly cite this article if you use the algorithm for your research.

Pre-requisites:

The following python packages must be present in the host machine before executing the main script.
> numpy

> scipy

> sklearn

> matplotlib

## Application Type 1

The software attempts to simulate a online/streaming feature scenario. First it builds a base model with a handful of features against which the new features is evaluated for goodness. Execute the following command:

> python demo_ofs.py


Input:

1. A data matrix with binary labels. The rows represent features and the columns represent samples. The first row contains the labels.
2. A linear classifier: Logistic Regression with a suitable parameter (we chose high lambda value to minimize regularizing effect)

>Output:

>>The following files are generated along with a figure:

>> 1. mfeat.init	 # a set of base features
>> 2. ent		 # evaluation score corresponding to each feature and sorted rank wise
>> 3. mfeat.entrank  # evaluation score corresponding to each feature
>> 4. mfeat.entauc   # evaluation score and improvement in AUC corresponding to each feature
>> 5. mfeat.contable # contingency table to perform statistical significance test

>>The script produces a figure that represents a contingency table. The fist quadrant demonstrates how the evaluated score correlates with the actual improvement. A statistical significance test  of the contingency table is also displayed in the output.

## Application Type 2

The same program can be used for simple feature selection avoiding over-fitting as well. Execute the following command:

> python demo_fs.py

Input:

1. A data matrix with binary labels. The rows represent features and the columns represent samples. The first row contains the labels.
2. A linear classifier: Logistic Regression with a suitable parameter (we chose high lambda value to minimize regularizing effect)
3. Initial set of base features, as generated from the first usage
4. Evaluation score corresponding to each feature, as generated from the first usage

> Output:

>> The script outputs model performance when feature subset is incremented through the batches of fixed size. Batch features are cumulatively added from the ranked subset produced by the proposed algorithm. The classifier is retrained every time the feature subset is incremented.


NB: All scripts have been tested on python version 2.6.
