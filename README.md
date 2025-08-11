# Can Machine Learning Beat the Odds?

## Introduction
In this project, I set out to explore a classic yet extremely challenging question: Is it possible to gain any kind of edge over the bookmakers in horse racing using machine learning models?
The motivation came from a curiosity to apply predictive modeling techniques to sports betting, and more specifically, horse racing in Hong Kong. However, I acknowledge from the outset the structural limitations of this approach:

The data used is outdated (up to 2005), meaning that markets today may be more efficient and odds potentially more tightly adjusted.

Bookmakers already incorporate a vast amount of information — including variables that are not available in the dataset used here.

The goal is not to prove that there’s a magical formula to beat the market, but rather to understand whether there are signs that a model can capture meaningful patterns not fully reflected in the odds.

This project should therefore be seen as a starting point: an initial attempt to model this type of problem, knowing that there is significant room for improvement whether through better feature engineering, more robust data handling, or integration with live odds and contextual information.

## Technical Summary of the Notebook
### 1. Data Preparation
The first step was to import and clean the data from the races and runs dataset, which both contains detailed information on horse races in Hong Kong. From the available data provided at https://www.kaggle.com/datasets/gdaley/hkracing I selected relevant columns such as: Horse, jockey, and trainer identifiers; Carried weight, draw position, previous rankings, ratings, and other attributes; Bookmaker odds (win_odds and pla_odds), which serve as market references.

Several important transformations were applied:

Type conversion (e.g., dates and categorical variables); Creation of a binary classification target: 1 if the horse won the race, 0 otherwise; Removal of rows with missing data and encoding of categorical variables using LabelEncoder; Generation of race_id to ensure that the models respected the race structure (i.e., multiple horses per race).

The dataset was then split into training and testing sets based on time to avoid leakage using older races for training and more recent ones for evaluation.

### 2. Model Training: Predicting the Winner
I used LightGBM as the main algorithm because it is efficient, robust with tabular data, and easy to calibrate. The model was trained to predict the probability of each horse winning — a classic imbalanced binary classification problem (only one winner per race). Other models were also tested (Logistic Regression, Random Forest, XGBoost), but LightGBM achieved better results in both log loss and betting simulations.

Importantly, I respected the race structure (i.e., no StratifiedShuffleSplit at the horse level) and ensured that predictions were made only with data available before the race.

### 3. Model based strategy vs Betting on favorites
We compare the model based strategy versus betting on the horse with the lowest win odds (market favorite).

### 4. Model-Based Betting Simulation
After generating the predicted probabilities (proba_model), I selected the horse with the highest probability for each race and simulated a unitary bet on it. Compared to the favorite strategy, the model produced better average returns per bet, especially when the favorite had higher odds (i.e., in more uncertain races).

I also tested:

Confidence filtering: only placing bets when proba_model ≥ threshold, with different thresholds, which reduced the number of bets but increased average return; Grouping by real odds: I observed that the model was more effective in races where the market was more divided (i.e., no clear favorite).

### 5. Experiment: "Placed" Bets (Top 3)
To try and improve hit rate, I modified the target to predict whether a horse would finish in the top 3. However, this strategy did not prove profitable:

The hit rate increased, but the odds for place bets were too low to compensate; False positives and overconfidence led to more consistent losses.

### 6. Model Comparison
I trained additional models beyond LightGBM:

Logistic Regression: fast and simple, but with lower predictive power; Random Forest: decent, but slower and with less generalization; XGBoost: similar to LightGBM but slightly inferior in results. LightGBM emerged as the most robust model in terms of both predictive accuracy and financial performance.

### 7. Visualizations and Metrics
Distribution plots of predicted probabilities (proba_model) and comparison with bookmaker odds; Return curves across confidence thresholds (e.g., performance when the model predicts ≥ 0.7); Basic statistics including total return, number of bets placed, and hit rate.

## Conclusion
After training several models and simulating betting strategies based on the predictions, I reached a few important conclusions:

LightGBM was the best-performing model overall, both in terms of predictive metrics and financial backtest results. Betting based on the model, particularly when applying a confidence threshold, outperformed simply betting on the market favorite in some scenarios. However, the margin remains small and far from guaranteeing consistent profits. I also tested “place” bets (finishing in the top 3), but that strategy did not prove to be profitable in this case.
Using confidence thresholds to filter bets (e.g., only betting if the predicted probability ≥ 0.7) helped improve the return per bet, although at the cost of fewer betting opportunities.
The most valuable insight from this exercise is that, even with limited data, the model was able to identify situations where the true probability of winning appeared underestimated by the market odds. This suggests that there may be potential to detect inefficiencies.
This work is not a ready-to-use system for real-world betting, but rather an exploratory foundation. I believe that, with the inclusion of new data, contextual variables (such as track condition, jockeys, betting flows, etc.), and more refined entry criteria improvements will be made.
