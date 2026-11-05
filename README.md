# midterm_team2_carprices

# Team 2 - Car prices and Economic Trends

## Project Overview
- For our midterm assessment project, we looked at how inflation and interest rates relate to car prices and sales between 2019 and 2023. We used a dataset from Kaggle with 60 months of data for the Toyota Camry in the USA. It included new and used car prices, inflation rates, interest rates and the amount of units sold. Python and Panda were used to clean the data. matplotlib was used to create graphs and regression models used to analyse the results.

## Instructions to Reproduce Results
-To run the project:
1. Clone the GitHub repository
2. Open it in Visual Studio Code with Python and Jupyter installed in the environment.
3. Install these using: pip install pandas numpy matplotlib scikit-learn jupyter
4. Open notebooks and then 01_data_exploration.ipynb and run the cells.
5. To run the regression models, open the terminal in the main project folder and enter `python modeling_analysis.py`. This will display the results for all four models.

## Summary of Findings
We first cleaned the dataset by removing one empty row and converting the prices, the sales figures and the rates into numbers. We then created three graphs to compare car prices, interest rates, inflation and sales.
- The first graph showed that new car prices gradually increased between 2019 and 2023. Used car prices decreased until around 2022 before starting to rise again.
- The second graph showed that higher interest rates were generally associated with higher car sales. This was unexpected, but it doesn't mean higher interest rates caused sales to increase.
- The third graph showed that used car prices generally tended to be lower when inflation was higher, while new car prices had a less obvious relationship.
We also compared four regression models being the Linear Regression, Polynomial Regression, k-NN and Decision Tree Regression. Linear Regression gave the best results in our model comparison.

Overall, the results showed different patterns in new and used car prices, but with only 60 monthly observations, we cannot confidently say which responded faster to economic changes or that interest rates directly caused changes in sales.
