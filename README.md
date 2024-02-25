# BoardGen: Moonboard Problem Classification and Generation   
**Overview**

BoardGen is a project focused on addressing the challenges of classifying and generating climbing problems on the Moonboard, a standardized climbing system with a vast array of user-created routes. This repository contains implementations of machine learning models and techniques to tackle these challenges.

**Machine Learning Classifier**

I developed a simple Multi-Layer Perceptron (MLP) model, leveraging extensive preprocessing techniques and dataset understanding. Surprisingly, the MLP model outperformed sophisticated deep learning models, including complex Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs), 
as well as surpassing the human benchmark for route classification. The above claim is made from the results described in the papers here: (https://cs230.stanford.edu/projects_spring_2020/reports/38850664.pdf) and here: (https://arxiv.org/pdf/2311.12419.pdf). Interestingly, this MLP performs significantly better
than the MLP mentioned in the papers, outperforming on both accuracy and +/- 1 accuracy (sadly, no F1 score was mentioned, and since the dataset is imbalanced accuracy doesn't mean much).   

**Problem Generator: Variational Autoencoder (VAE)**

Additionally, I implemented a Variational Autoencoder (VAE) capable of generating climbing problems, with the ability to tweak the number of holds included in each problem. This was done by treating the generated "board" as a probabalistic map of holds, and creating a board based on the top x number of holds in a given board map. This approach enriches the Moonboard experience by generating diverse and engaging climbs.
Repository Structure

    grade_classifier.ipynb: Contains the implementation of the Multi-Layer Perceptron (MLP) model for route classification.
    VAE-route-gen.ipynb: Includes the code for the Variational Autoencoder (VAE) used to generate climbing problems.
    preprocessor.ipynb: Scripts and notebooks for preprocessing the Moonboard dataset.
    /Web_Interface: Web interface implementation for interacting with the models. (TODO)

Usage
Requirements

    Python 3.x
    TensorFlow
    NumPy
    Pandas
    Matplotlib

Contribution

Contributions to improve the project are welcome! Feel free to fork this repository, make your enhancements, and create a pull request.
References
