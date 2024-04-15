# BoardGen: Moonboard Problem Classification and Generation   
**Overview**

BoardGen is a project focused on addressing the challenges of classifying and generating climbing problems on the Moonboard, a standardized climbing system with a vast array of user-created routes. This repository contains implementations of machine learning models and techniques to tackle these challenges.

**Machine Learning Classifier**

I developed a simple Multi-Layer Perceptron (MLP) model, leveraging extensive preprocessing techniques and dataset understanding, along with the implementation of CORAL-style ordinal regression ideas. Surprisingly, the MLP model outperformed the sophisticated deep learning models found online, including complex Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs),  as well as surpassing the human benchmark for route classification. The above claim is made from the results described in the papers here: (https://cs230.stanford.edu/projects_spring_2020/reports/38850664.pdf) and here: (https://arxiv.org/pdf/2311.12419.pdf). Interestingly, this MLP performs significantly better than the MLP mentioned in the papers, outperforming on both accuracy (~50%) and +/- 1 accuracy (~90%) (sadly, no F1 score was mentioned, and since the dataset is imbalanced accuracy doesn't mean much). It still struggles on grades above V8, since the examples there are very limited, but overall performance is the best that I could find online. The classifier was then built into an ONXX compatible network, hosted on a web application using fly at https://moonboard-grade-predict.fly.dev/

Additionally, due to the way I constructed the data, assuming the moonboard as an "image" of the board, with different channels corresponding to different types of holds, two additional models were built to try and better leverage the spatial properties within the data. These models were a simple shallow CNN, and a deeper ResNet style network, where skip connections allows for deep networks without vanishing or exploding gradients. In addition to these new models, slight label smoothing was used as an enhancement to the training loop, since the labels are noisy. This label smoothing was done with an (as far as I could find) original method, where the boundaries of ordinal labels are smoothed more significantly than non-boundary labels, providing a more generalized model that more smoothly interprets between ordinal labels. The performance comparison is below (on the testing set), with the best performers in bold for each category:

| Model             | Exact Acc. | +-1 Acc. | Macro F1 Score | Weighted F1 Score |
| :---------------: | :--------: | :------: | :------------: | :----------------:
| MLP               |  55.88%    | 91.27%   | 0.3980         | 0.5631
| Simple CNN        |  56.05%    | **92.13%** | 0.4020       | 0.5706
| ResNet            |**58.60%**  | 91.30%   | **0.4052**     | **0.5765**

**Problem Generator: Wasserstein Conditional Generative Adversarial Network with Gradient Penalty and Diversity Regularization (W-CGAN-GP-DR)**

Additionally, I implemented a deep Conditional Generative Adversarial Network (CGAN) capable of generating climbing problems, with the ability to tweak the number of holds included in each problem. This was done by treating the generated "board" as a probabalistic map of holds, and creating a board based on the top x number of holds in a given board map. This approach enriches the Moonboard experience by generating diverse and engaging climbs. Since the dataset has imbalanced classes, and training a Conditional GAN is very difficult and prone to mode collapse already, a number of methods were used to mitigate this issue. Firstly, the Wasserstein Distance with Gradient Penalty method was used for training instead of the canonical Discriminator approach, providing more stability during training without vanishing gradients. Further, during training noise was added to the discrete "board" examples so as to increase the stability of the gradient calculations and allow for easier learning. Finally, to prevent mode collapse, a Diversity based regularization method was used, similar in idea to the one outlined in the paper "Diversity Regularized Adversarial Learning" by Babajide O. Ayinde, Keishin Nishihama, and Jacek M. Zurada, using L1 norms instead of eucliean norms, as they may perform better in high dimensional distance space. The result of all of these additions to the training allowed for stability and diverse output, with relatively okay quality problems. If I were to do this again, I would use some sort of recurrent formulation for this, but I both wanted to try my hand at GANs and wanted to differentiate from the previous approaches to this problem, and so I am still happy with my choice to go with GANs.   
**Repository Structure**

    grade_classifier.ipynb: Contains the implementation of the Multi-Layer Perceptron (MLP) model for route classification.   
    W-CGAN-GP-Board-Gen.ipynb: Includes the code for the GAN used to generate climbing problems.   
    preprocessor.ipynb: Scripts and notebooks for preprocessing the Moonboard dataset.   
    /Web_Interface: Web interface implementation for interacting with the models. (TODO)   

Usage
Requirements

    Python 3.x
    PyTorch
    NumPy
    Pandas
    Matplotlib
    Seaborn

Contribution

Contributions to improve the project are welcome! Feel free to fork this repository, make your enhancements, and create a pull request.   

References   
https://cs230.stanford.edu/projects_spring_2020/reports/38850664.pdf   
https://arxiv.org/pdf/2311.12419.pdf   
https://arxiv.org/pdf/1901.10824.pdf   
