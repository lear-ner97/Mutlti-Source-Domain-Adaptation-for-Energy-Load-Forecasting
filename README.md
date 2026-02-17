# Mutlti-source-domain-adaptation-for-energy-load-forecasting
This is the official implementation of our submitted research paper "An explainable unified multi-source domain adaptation  framework for short-term energy load forecasting"

# Main Contribution
implementation of an explainable hybrid multi-source domain adaptation model to improve the prediction accuracy of daily energy load in a data-scarce domain

## Step-by-step guidelines to reproduce the results
The repo's code is implemented for a four-source domain adaptation forecaster. If you are interested in reproducing the results of the paper, please follow the guidelines below.

## Install dependencies
put all the github repo in a single folder, set your project directory, and install the packages listed in main_multi_source_implementation.py/lines 8-18 and SHAP.py/ lines 10-17

## multi-source model setup
1-you should modify the values of the lookback window T and the future horizon H in lines 95-98 based on the type of forecasting:<br>
-hourly: T=24 and H=1.<br>
-daily: T=24*7 and H=24.<br>
2-for the source data train-validation-test split, please refer to lines 200-206. <br>
3-The values of certain hyperparameters depend on the forecasting scenario (hourly or daily). Please, refer to lines 320 to 350 and follow the comments at each line to set the right values. <br>
4-For reproducibility of the results, you have to modify these lines:<br>
*main_multi_source_implementation.py/line 383, you should adjust the number of source features depending on how many source datasets you are using. For example, if you use three source datasets, you have to remove src4_features.<br>
*functions.py: you have to adjust the following lines, depending on the number of source datasets you want to use:<br>
  - line 214: the number of data loaders and their indices.<br>
  - lines 219 to 222: uncomment/comment the batches based on the number of source datasets. For instance, if you are using two source datasets you comment src3 and src4 batches. 
  - lines 227 to 257: you uncomment/comment the blocks, based on the number of sources. For instance, if you use three source datasets, you uncomment the final block.<br>
  - line 261: you remove the unused final features.<br>
  - line 289: you return the final features, based on how many sources you are using.

## Training
run the file main_multi_source_implementation.py.  <br>    The expected output: <br>  1/ test metrics  <br> 2/plot of test prediction vs true data<br>
PS: If you want to train the target-only model (without domain adaptation), you just comment lines 383-385 and uncomment lines 387-389.

## Description of each file (ordered in alphabetical order)
*SHAP.py: once you finish the model training, you use the code in shap.py to visualize shap explanation graphs.<br>
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
*normalized_average_daily_load.py: analysis of the average daily load of different buildings on weekends, weekdays, and correlation analysis of temperature-load relationships.<br>
*plot_models_prediction.py: used to plot the test prediction vs true data, with different number of source domains.

## 📬 Contact
If you have questions or encounter issues, please [open an issue](https://github.com/lear-ner97/Mutlti-Source-Domain-Adaptation-for-Energy-Load-Forecasting/issues) or contact us at **sami DOT benbrahim AT mail DOT concordia DOT ca**.


