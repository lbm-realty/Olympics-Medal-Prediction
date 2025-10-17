Overview

A linear regression model that predicts how many medals a country will win basesd on previous years' performance and the number of athletes it entered.
    -> Previous wins matter because they give us an idea of a country's general preformance
    -> Number of athletes matters because the higher the number of athletes a country enters, the more likely it is for it win more medals

Steps
    
    * Data Preparation:
        -> Loaded data from teams.csv.
        -> Kept only key columns: team, country, year, athletes, age, prev_medals, and medals.
        -> Encoded text columns (country, team) into numbers.
        -> Removed missing values.

    * Model Training:
        -> Used data before 2012 for training and 2012+ for testing.
        -> Trained a Linear Regression model with predictors athletes and prev_medals.

    * Evaluation:
        -> Compared predictions to actual medals.
        -> Measured accuracy using Mean Absolute Error (MAE).
        -> Visualized relationships and error distribution.

    * Output
        -> The model predicts medal counts for each team and shows how accurate those predictions are across different countries.