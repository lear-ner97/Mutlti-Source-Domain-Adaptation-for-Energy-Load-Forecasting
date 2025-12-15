# Mutlti-source-domain-adaptation-for-energy-load-forecasting
This is the official implementation of our submitted research paper "An explainable unified multi-source domain adaptation  framework for short-term energy load forecasting"

# Main Contribution
implementation of an explainable hybrid multi-source domain adaptation model to improve the prediction accuracy of daily energy load in a data-scarce domain

## Step-by-step guidelines to reproduce the results
The repo's code is implemented for a four-source domain adaptation forecaster. If you are interested in reproducing the results of the paper, please follow the guidelines below.

## Install dependencies
put all the github repo in a single folder, set your project directory, and install the packages listed in main_multi_source_implementation.py/lines 8-18

## multi-source model setup
No need to change the values of hyperparameters. <br>
For reproducibility of the results, you have to modify these lines:<br>
*main_multi_source_implementation.py/line 378, you should adjust the number of source features depending on how many source datasets you are using. For example, if you use three source datasets, you have to remove src4_features.<br>
*functions.py: you have to adjust the following lines, depending on the number of source datasets you want to use:<br>

  -line 214: the number of data loaders and their indices.<br>
  - lines 227 to 257: you uncomment/comment the blocks, based on the number of sources. For instance, if you use three source datasets, you uncomment the final block.<br>
  - line 261: you remove the unused final features.<br>
  - line 289: you return the final features, based on how many sources you are using.

## Training
run the file main_multi_source_implementation.py.  <br>    The expected output: <br>  1/ test metrics  <br> 2/plot of test prediction vs true data<br>
PS: If you want to train the target-only model (without domain adaptation), you just comment lines 376-380 and uncomment lines 382-384.

## Description of each file
*shap.py: once you finish the model training, you use the code in shap.py to visualize shap explanation graphs.<br>
*attention.py: once you finish the model training, you use the code in attention.py to visualize the attention weights of the model.<br>
*boxplot_mape.py: after you train the model with 10 random seeds, with all possible number of sources, you use the resulting test mape values to plot the boxplots.<br>
*boxplot_rmse.py and boxplot_mae.py: same as boxplot_mape.py but for different metrics.<br>
*data_collection_and_cleaning.py: data collection and cleaning.<br>
*data_visualization.py: data visualization.<br>
*error_distribution.py: plot of model's test error distribution.<br>
*functions.py: contains the training function, data loaders, model, mmd function.<br>
*histograms.py: plots the data histogram.<br>
*main_source_selection.py: used for source domain selection.<br>
*main_multi_source_implementation.py: multi-source domain model training.<br>
*plot_models_prediction.py: used to plot the test prediction vs true data, with different number of source domains.

## 📬 Contact
If you have questions or encounter issues, please [open an issue](https://github.com/lear-ner97/Mutlti-Source-Domain-Adaptation-for-Energy-Load-Forecasting/issues) or contact us at ** sami DOT benbrahim AT mail DOT concordia DOT ca **.


