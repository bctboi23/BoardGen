# BoardGen: Moonboard Problem Classification and Generation   
**Overview**

BoardGen is a project focused on addressing the challenges of classifying and generating climbing problems on the Moonboard, a standardized climbing system with a vast array of user-created routes. This repository contains implementations of machine learning models and techniques to tackle these challenges.

**Machine Learning Classifier**

I developed a simple Multi-Layer Perceptron (MLP) model, leveraging extensive preprocessing techniques and dataset understanding, along with the implementation of CORAL-style ordinal regression ideas. Surprisingly, the MLP model outperformed sophisticated deep learning models, including complex Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs),  as well as surpassing the human benchmark for route classification. The above claim is made from the results described in the papers here: (https://cs230.stanford.edu/projects_spring_2020/reports/38850664.pdf) and here: (https://arxiv.org/pdf/2311.12419.pdf). Interestingly, this MLP performs significantly better
than the MLP mentioned in the papers, outperforming on both accuracy (~50%) and +/- 1 accuracy (~90%) (sadly, no F1 score was mentioned, and since the dataset is imbalanced accuracy doesn't mean much). It still struggles on grades above V8, since the examples there are very limited, but overall performance is the best that I could find online.  
The classifier was then built into an ONXX compatible network, hosted on a web application using fly at https://moonboard-grade-predict.fly.dev/

Additionally, due to the way I constructed the data, assuming the moonboard as an "image" of the board, with different channels corresponding to different types of holds, two additional models were built to try and better leverage the spatial properties within the data. These models were a simple shallow CNN, and a deeper ResNet style network, where skip connections allows for deep networks without vanishing or exploding gradients. The performance comparison is below, with the best performers in bold for each category:

| Model             | Exact Acc. | +-1 Acc. | Macro F1 Score | Weighted F1 Score |
| :---------------: | :--------: | :------: | :------------: | :----------------:
| MLP               |  53.12%    | 92.23%   | 0.3799         | 0.5457
| Simple CNN        |  54.35%    | **93.36%** | 0.3992       | 0.5574
| ResNet            |**58.36%**  | 92.81%   | **0.4311**     | **0.5778**

**Problem Generator: Variational Autoencoder (VAE)**

Additionally, I implemented a Variational Autoencoder (VAE) capable of generating climbing problems, with the ability to tweak the number of holds included in each problem. This was done by treating the generated "board" as a probabalistic map of holds, and creating a board based on the top x number of holds in a given board map. This approach enriches the Moonboard experience by generating diverse and engaging climbs.   
**Repository Structure**

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
