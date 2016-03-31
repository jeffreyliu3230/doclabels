# doclabels

## Classify SHARE documents with Natural Language Processing

Jiankun Liu, 03/22/2016

Developers at the Center for Open Science working on the [SHARE](http://osf.io/share) project are constantly looking for ways to improve SHARE’s metadata quality. One challenging task is to add subject areas so that users can have more options and control when searching and filtering documents. Since we have more than 5 million documents on SHARE, manually labeling the documents would be very tough. Therefore, we need to rely on an automated process that can achieve fairly high precision. That’s where machine learning comes in.

To tackle the problem, I built the [multi-label document classification](https://en.wikipedia.org/wiki/Multi-label_classification) model using training data from the [Public Library of Science](http://plos.org) API, which stores more than 160 thousand documents with subject areas defined by their [taxonomy](http://www.plosone.org/taxonomy). The documents contain titles and abstracts that can be used to generate features for the classification model. The taxonomy has a hierarchical structure that contains more than ten thousands terms, but only the terms on the root are used as the subjects areas in our model to begin with. Documents are cleaned and normalized before being stored in the the database (MongoDB).

![plos](http://i2.wp.com/www.highdimensional.space/wp-content/uploads/2016/03/plos.png?resize=640%2C426)

These documents provide abundant yet [imbalanced](http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) training data for our supervised machine learning model (for example more than 90 percent documents are labeled “Biological and life sciences”, which badly affects the predictions). A lot of preprocessing methods were used to address multiple issues in the dataset which will be illustrated in a follow-up post, but here is a simplified workflow:

![flow1](http://i0.wp.com/www.highdimensional.space/wp-content/uploads/2016/03/flow1.png?resize=640%2C411)

To begin with, the documents are transformed into bag-of-words vector representations by calculating the [n-gram](https://en.wikipedia.org/wiki/N-gram) term frequency and [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (term frequency – inverse document frequency) of each term. The features are used to fit the classifiers. Since this is a multi-label classification problem (each document can have multiple subject areas), we trained 11 [One-vs-rest classifiers](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest), where each classifier was exclusively used to identify if one document belongs to one particular subject area. For example, when training the “Earth sciences” classifier, all documents that have “Earth sciences” as one of their subject areas will be labeled 1 and others labeled 0\. By training those classifiers separately it provided more flexibility for tuning so that we can deploy the models with good precision while keep improving other ones. The best classifiers could achieve over 90 percent [precision](https://en.wikipedia.org/wiki/Precision_and_recall) (number of true-positive over all documents predicted as positive, which we cared the most in this case), while others need further optimization. Nevertheless we are confident the model will keep improving over time with more feature engineering (e.g. adding [Word2vec](https://en.wikipedia.org/wiki/Word2vec)), more diverse training data, and more parameter optimization.

Finally, we need to take into consideration the scalability of our framework. Traditional methods described above requires all training data to be loaded in the memory at once. In case of increasing training data size, I built a framework that can utilize batch training methods (or [online learning](https://en.wikipedia.org/wiki/Online_machine_learning)) and feed in data one chunk at a time:

![train1](http://i2.wp.com/www.highdimensional.space/wp-content/uploads/2016/03/train1.png?resize=640%2C170)

A follow-up post will further explain the detailed preprocessing, feature engineering, and modeling steps. As a bonus we will also show how to use Google’s [TensorFlow](https://www.tensorflow.org/) to build a [Convolutional Neural Network](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/) for the text classification problem.

Thanks to Katherine Schinkel who contributed to model selection and metrics.
