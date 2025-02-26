# MH6805 Script Video Recording

Hey everyone,

I’ll be continuing our presentation on our article:
Bitcoin Price Prediction Using Machine Learning.

Previously, we introduced Bitcoin and discussed the importance of price prediction using ML techniques.

Now, we’re going to dive deeper into this topic by exploring the methodologies we used and the results we obtained from our analysis.

When it comes to methodology, one of the most crucial steps is choosing the right model.
Which models should we use?
Why did we choose them?
And what do we expect from their implementation and analysis?

For our study, we selected two models: XGBoost and SVR.
But how do we decide on the right model?
Which one should we ultimately rely on?

To answer that, we need to take a closer look at these models and explain why they are valuable in solving our problem.

XGBoost excels in handling large datasets, capturing non-linear relationships, and providing feature importance, making it ideal for Bitcoin price prediction with complex market data. SVR (Support Vector Regression) is powerful for small to medium datasets, effectively modeling non-linear trends while maintaining robust generalization, making it useful for short-term BTC price forecasting with fewer features.

## How XGBoost Works + Hyperparameter Tuning

## How XGBoost Works: Step-by-Step

    Initial Prediction:
        XGBoost starts with a simple prediction, often the mean of the target variable (e.g., Bitcoin price).

    Creating Decision Trees:
        It builds small decision trees, where each tree learns from previous mistakes.

    Boosting Process:
        New trees focus more on hard-to-predict cases, improving accuracy iteratively.

    Regularization:
        It prevents overfitting by using L1/L2 penalties and controlling tree depth.

    Final Prediction:
        All trees’ outputs are combined, giving a stronger, more accurate final prediction.


## How SVR (Support Vector Regression) Works + Hyperparameter Tuning


## How SVR (Support Vector Regression) Works: Step-by-Step

    Defines an "ε-insensitive" Margin:
        SVR ignores errors within a certain margin (ε), only penalizing larger deviations.

    Uses Support Vectors:
        It selects key data points (support vectors) that define the best-fit hyperplane.

    Works with Kernel Functions:
        SVR maps features into higher dimensions (e.g., polynomial, RBF kernels) to model non-linear trends, making it effective for Bitcoin price prediction in volatile markets.

# Prediction Model

Model Performance (MAE, MSE, RMSE) 


Let's see the graphics:

This XGBoost model predicts Bitcoin prices with a Mean Absolute Error (MAE) of 1,604, indicating a relatively small average deviation. The Root Mean Squared Error (RMSE) of 2,062 suggests decent accuracy, closely tracking actual prices, though some deviations are visible.

The Mean Squared Error (MSE) of 4,253,529 is relatively high, indicating larger squared deviations in some predictions. However, since MSE heavily penalizes large errors, the lower RMSE (2,062) suggests that overall prediction performance is still reasonably accurate.

This SVR model achieves a lower MAE (659) and RMSE (988) compared to XGBoost, indicating more precise predictions. The MSE (977,221) is significantly lower, suggesting fewer large errors. The model closely tracks actual Bitcoin prices with minimal deviations.

Model Selection Matters – The comparison between XGBoost and SVR shows that SVR performed better in this case, with lower errors (MAE: 659 vs. 1,604 and RMSE: 988 vs. 2,062), meaning it provided more accurate predictions.


# Key Feature Importance in Models

Feature Importance Is Key – The most influential factors in predicting Bitcoin prices were NASDAQ Close, BTC Open, and Lower_B (likely a technical indicator). This suggests that Bitcoin’s price movements are closely tied to traditional financial markets.

This SVR model identifies EMA_10, BTC High, and BTC Open as the most critical features for Bitcoin price prediction. This suggests that recent trends (EMA), daily highs, and opening prices play a crucial role in determining future price movements.

# Comparison of Model Results (XGBoost vs. SVR)

Key Conclusions from Bitcoin Price Prediction

    Model Selection Matters – The comparison between XGBoost and SVR shows that SVR performed better in this case, with lower errors (MAE: 659 vs. 1,604 and RMSE: 988 vs. 2,062), meaning it provided more accurate predictions.

    Feature Importance Is Key – The most influential factors in predicting Bitcoin prices were NASDAQ Close, BTC Open, and Lower_B (likely a technical indicator). This suggests that Bitcoin’s price movements are closely tied to traditional financial markets.

    Machine Learning Can Reduce Uncertainty, Not Eliminate It – While models provide valuable insights, Bitcoin’s price remains highly volatile, and external factors (regulations, macroeconomic events, investor sentiment) can create deviations that even the best models may struggle to predict.

    Lower Errors, But Not Perfect Predictions – The errors are relatively low, but not insignificant. A lower RMSE indicates a good model fit, but large price swings can still lead to unpredictable movements.

    Predicting Bitcoin Is Challenging Due to Market Dynamics – Unlike stocks, Bitcoin is influenced by whale movements, market manipulation, and social sentiment, which are harder to model using historical price data alone.

Final Takeaway:

Machine learning models can improve Bitcoin price forecasting but should not be solely relied upon for financial decisions. Combining ML predictions with fundamental analysis, market news, and risk management strategies is crucial for better-informed decisions.
