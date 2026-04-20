# Model Training Steps

## 1. Standard Steps of EDA:
  #### a. Import modules (sklearn, pandas, numpy, xgboost).
  #### b. Define the Pandas dataframe with the csv filename.
  #### c. Get the head, info and description of the dataframe.
  #### d. Look for any missing row values.
  #### e. Drop the irrelevant columns such as `id, vendor_id, store_and_fwd_flag`

## 2. Feature Engineering Methods
  #### a. Restrict the df to only hold non zero passenger rides less than 6 to take care of outliers.
  #### b. Extract usable info from the `pickup_datetime` column.
  #### c. Get new columns `pickup_hour`, `pickup_dow`, `pickup_month` to use later on.
  #### d. Use the `pickup_lat`. `pickup_long` and `dropoff_lat`, `dropoff_long` to calculate the Haversine Distance between the pickup and dropoff points.
  #### e. Restrict the latitude and longitude values to the geographical limits of NYC
  #### f. Round the lat and long values to 2 decimal places to group areas into rough 1x1 km grids (Useful in extracting the zonal data if required).
  #### g. Drop the original lat and long columns after inserting the rounded values to new columns.
  #### h. Apply `Cyclic Encoding` to `pickup_hour`, `pickup_dow` and `pickup_month` to preserve the cyclic nature. This ensures that boundary values (e.g., 23 → 0 hours, Sunday → Monday, December → January) are treated as adjacent rather than numerically distant, allowing the model to learn meaningful temporal patterns.
  #### i. Create new columns `distance_hour`, `distance_dow`,	`distance_month` by multiplying the distance and respective cyclic value columns. This step defines the relevance of the time that the trip occurs.
  #### j. Drop the rows having `trip_duration` less than 60 seconds and more than 7200 seconds and sort the df.

## 3. Training Steps
  #### a. A chronological split is applied so that the model learns from the first 80% of the data, and then tested against the last 20%. Make 2 seperate dataframes for training and testing.
  #### b. Define the variables `X_train`, `X_test`, `y_train` and `y_test` on the training and testing dataframes respectively, ensuring that the target and feature columns are kept in mind.
  #### c. Define the pipeline with the model of your choice. `XGBRegressor` is used in this case.
  #### d. Define a `param_grid`. This lets you tune the hyperparameters to minimize the error in predictions.
  #### e. Set up a `GridSearchCV` grid to find the optimal combination of hyperparameters to improve model performance.
  #### f. Then use `grid.fit` and `best_model = grid.best_estimator_` 
  #### g. Compute the RMSE and compare it against the baseline RMSE.

## 4. Build the Model
  #### a. Import the pickle model and dump the trained model to a .pkl file for further progress.
  