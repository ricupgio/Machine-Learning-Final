import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


import re 

from datetime import timedelta

final_test = pd.read_csv('final_test.csv')
train = pd.read_csv('final_train.csv')

train.hist(bins = 50, figsize = (20,12), xlabelsize =0.8)

numeric_columns = train.select_dtypes(include=np.number)
corr_price = numeric_columns.corr()["price"].sort_values(ascending=False)

#split train set into two parts so we can test the data
train_set, test_set = train_test_split(train, test_size = 0.2, random_state = 42)

#find what variables are missing
count_missing_train = train_set.isnull().sum().sort_values(ascending= False)
percent_missing_train = ((train_set.isnull().sum())/(train_set.isnull().count())*100).sort_values(ascending = False)
percent_missing_train

"""
missing:
segmentsEquipmentDescription         38.735714
totalTravelDistance                  13.558929
segmentsCabinCode                    12.500000
segmentsDistance                      7.274107
seatsRemaining                        1.250000
"""
#dont think the type of plane will affect prices, also too much missing data
train_set.drop(['segmentsEquipmentDescription'],axis = 1, inplace=True)

#since I am already using the total distance
# I dont believe segments distance will be that helpful

train_set.drop(['segmentsDistance'],axis = 1, inplace=True)

#start with total travel distance

## Function to convert travel duration string to numeric value in minutes
def convert_duration(duration_str):
    duration = timedelta()
    # Use regular expression to extract hours and minutes
    match = re.match(r'PT(\d+)H(\d+)M', duration_str)
    if match:
        hours,minutes = map(int, match.groups())
        duration = timedelta(hours = hours, minutes = minutes)
    else:
        match = re.match(r'PT(\d+)H', duration_str)
        if match:
            hours = int(match.group(1))
            duration = timedelta(hours = hours)
    total_minutes = duration.total_seconds()//60
    return int(total_minutes)

travel_duration_column = 'travelDuration'

# Apply the conversion function to the 'travelDuration' column
train_set[travel_duration_column] = train_set[travel_duration_column].apply(convert_duration)

total_travel_time_column = 'travelDuration'
total_travel_distance_column = 'totalTravelDistance'

# Calculate average speed using available data
average_speed = train_set[total_travel_distance_column].mean() / train_set[total_travel_time_column].mean()

# Identify rows with missing total travel distance
missing_distance_rows = train_set[train_set[total_travel_distance_column].isnull()]

# Impute missing distances based on average speed
train_set.loc[missing_distance_rows.index, total_travel_distance_column] = (
    missing_distance_rows[total_travel_time_column] * average_speed
)

#now have to do the missing cabin codes
#see what's in the data
cabinCodes = train_set['segmentsCabinCode'].unique()

cabin_code_column = 'segmentsCabinCode'

#last missing column
# Mode imputation
mode_cabin_code = train_set[cabin_code_column].mode()[0]
train_set[cabin_code_column].fillna(mode_cabin_code, inplace=True)

seats_remaining_column = 'seatsRemaining'

# Mean imputation
mean_seats_remaining = train_set[seats_remaining_column].mean()
train_set[seats_remaining_column].fillna(mean_seats_remaining, inplace=True)

#make new column showing days between searched and flight date
date_column1 = 'flightDate'  # Replace with the actual column name
date_column2 = 'searchDate'  # Replace with the actual column name

# Convert date strings to datetime objects
train_set[date_column1] = pd.to_datetime(train_set[date_column1], format='%Y-%m-%d')
train_set[date_column2] = pd.to_datetime(train_set[date_column2], format='%Y-%m-%d')

# Create a new column for the day difference
train_set['dayDifference'] = (train_set[date_column1] - train_set[date_column2]).dt.days

#create new column to track number of connections
train_set['connections'] = train_set['segmentsArrivalAirportCode'].apply(lambda x: x.count('||'))

train_num = train_set.select_dtypes(include=np.number)
train_cat = train_set.select_dtypes(exclude=np.number)

skewness = train_num.skew().sort_values(ascending=False)

skewed_vars = skewness.index[skewness >= 1].to_list()

#dont want to change the given price data
skewed_vars.remove('price')

#split into skewed and unskewed
unskewed_num_data = train_num.drop(skewed_vars, axis=1)
skewed_num_data = train_num[skewed_vars]

adjust_skew = np.log1p(skewed_num_data)

train_num = pd.concat([unskewed_num_data, adjust_skew], axis=1)

train_cat = train_cat.astype('category')

airport = train_cat['startingAirport'].unique()

airport_mapping = {'OAK': 1, 'SFO': 2, 'ORD': 3, 'MIA': 4, 'LAX': 5,
                   'EWR': 6, 'LGA': 7, 'JFK': 8, 'ATL': 9, 'DFW': 10,
                   'DEN': 11, 'BOS': 12, 'IAD': 13, 'CLT': 14, 'DTW': 15,
                   'PHL': 16}

starting_airport_column = 'startingAirport'
destination_airport_column = 'destinationAirport'


# Replace airport codes with numbers
train_cat[starting_airport_column] = train_cat[starting_airport_column].replace(airport_mapping)
train_cat[destination_airport_column] = train_cat[destination_airport_column].replace(airport_mapping)

#remove all spaces
train_cat[cabin_code_column] = train_cat[cabin_code_column].str.replace(' ','')

# Split the cabin codes based on '||'
train_cat[cabin_code_column] = train_cat[cabin_code_column].str.replace('||',' ')

train_cat[cabin_code_column] = train_cat[cabin_code_column].str.split()

# Create a mapping dictionary for individual cabin codes to numbers
individual_cabin_mapping = {'coach': 1, 'premiumcoach': 2, 'business': 3,'first': 4  }
# Replace individual cabin codes with numbers
train_cat['cabinCodeNums'] = train_cat[cabin_code_column].apply(lambda x: [individual_cabin_mapping.get(code, code) for code in x])

# Calculate the average
train_cat["averageCabinCode"] = train_cat['cabinCodeNums'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

num_from_cat_vars = ['startingAirport','destinationAirport', 'averageCabinCode']

num_from_cat = train_cat[num_from_cat_vars]
num_from_cat = num_from_cat.astype(int)

train_cat = train_cat.drop(num_from_cat,axis = 1)
train_cat = train_cat.drop(['segmentsCabinCode','cabinCodeNums'],axis = 1)

train_set_allnum = pd.concat([train_num, num_from_cat, train_cat['id']],axis = 1)

#now do the same thing for the test set
#dont think the type of plane will affect prices, also too much missing data
test_set.drop(['segmentsEquipmentDescription'],axis = 1, inplace=True)

#since I am already using the total distance
# I dont believe segments distance will be that helpful

test_set.drop(['segmentsDistance'],axis = 1, inplace=True)

#start with total travel distance

travel_duration_column = 'travelDuration'

# Apply the conversion function to the 'travelDuration' column
test_set[travel_duration_column] = test_set[travel_duration_column].apply(convert_duration)


# Replace 'totalTravelTimeSeconds' and 'totalTravelDistance' with the actual column names in your dataset
total_travel_time_column = 'travelDuration'
total_travel_distance_column = 'totalTravelDistance'

# Calculate average speed using available data
average_speed = test_set[total_travel_distance_column].mean() / test_set[total_travel_time_column].mean()

# Identify rows with missing total travel distance
missing_distance_rows = test_set[test_set[total_travel_distance_column].isnull()]

# Impute missing distances based on average speed
test_set.loc[missing_distance_rows.index, total_travel_distance_column] = (
    missing_distance_rows[total_travel_time_column] * average_speed
)

#now have to do the missing cabin codes
#see what's in the data
cabinCodes = test_set['segmentsCabinCode'].unique()

cabin_code_column = 'segmentsCabinCode'

#last missing column
# Mode imputation
mode_cabin_code = test_set[cabin_code_column].mode()[0]
test_set[cabin_code_column].fillna(mode_cabin_code, inplace=True)

seats_remaining_column = 'seatsRemaining'

# Mean imputation
mean_seats_remaining = test_set[seats_remaining_column].mean()
test_set[seats_remaining_column].fillna(mean_seats_remaining, inplace=True)

#make new column showing days between searched and flight date
date_column1 = 'flightDate'  # Replace with the actual column name
date_column2 = 'searchDate'  # Replace with the actual column name

# Convert date strings to datetime objects
test_set[date_column1] = pd.to_datetime(test_set[date_column1], format='%Y-%m-%d')
test_set[date_column2] = pd.to_datetime(test_set[date_column2], format='%Y-%m-%d')

# Create a new column for the day difference
test_set['dayDifference'] = (test_set[date_column1] - test_set[date_column2]).dt.days

#create new column to track number of connections
test_set['connections'] = test_set['segmentsArrivalAirportCode'].apply(lambda x: x.count('||'))

test_num = test_set.select_dtypes(include=np.number)
test_cat = test_set.select_dtypes(exclude=np.number)

skewness = test_num.skew().sort_values(ascending=False)

skewed_vars = skewness.index[skewness >= 1].to_list()

#dont want to change the given price data
skewed_vars.remove('price')

#split into skewed and unskewed
unskewed_num_data = test_num.drop(skewed_vars, axis=1)
skewed_num_data = test_num[skewed_vars]

adjust_skew = np.log1p(skewed_num_data)

test_num = pd.concat([unskewed_num_data, adjust_skew], axis=1)

test_cat = test_cat.astype('category')

airport = test_cat['startingAirport'].unique()

airport_mapping = {'OAK': 1, 'SFO': 2, 'ORD': 3, 'MIA': 4, 'LAX': 5,
                   'EWR': 6, 'LGA': 7, 'JFK': 8, 'ATL': 9, 'DFW': 10,
                   'DEN': 11, 'BOS': 12, 'IAD': 13, 'CLT': 14, 'DTW': 15,
                   'PHL': 16}

starting_airport_column = 'startingAirport'
destination_airport_column = 'destinationAirport'


# Replace airport codes with numbers
test_cat[starting_airport_column] = test_cat[starting_airport_column].replace(airport_mapping)
test_cat[destination_airport_column] = test_cat[destination_airport_column].replace(airport_mapping)

#remove all spaces
test_cat[cabin_code_column] = test_cat[cabin_code_column].str.replace(' ','')

# Split the cabin codes based on '||'
test_cat[cabin_code_column] = test_cat[cabin_code_column].str.replace('||',' ')

test_cat[cabin_code_column] = test_cat[cabin_code_column].str.split()

# Create a mapping dictionary for individual cabin codes to numbers
individual_cabin_mapping = {'coach': 1, 'premiumcoach': 2, 'business': 3,'first': 4  }
# Replace individual cabin codes with numbers
test_cat['cabinCodeNums'] = test_cat[cabin_code_column].apply(lambda x: [individual_cabin_mapping.get(code, code) for code in x])

# Calculate the average
test_cat["averageCabinCode"] = test_cat['cabinCodeNums'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

num_from_cat_vars = ['startingAirport','destinationAirport', 'averageCabinCode']

num_from_cat = test_cat[num_from_cat_vars]
num_from_cat = num_from_cat.astype(int)

test_cat = test_cat.drop(num_from_cat,axis = 1)
test_cat = test_cat.drop(['segmentsCabinCode','cabinCodeNums'],axis = 1)

test_set_allnum = pd.concat([test_num, num_from_cat, test_cat['id']],axis = 1)


# Drop unnecessary columns
final_test.drop(['segmentsEquipmentDescription'], axis=1, inplace=True)
final_test.drop(['segmentsDistance'], axis=1, inplace=True)

# Apply the conversion function to the 'travelDuration' column
final_test[travel_duration_column] = final_test[travel_duration_column].apply(convert_duration)

# Calculate average speed using available data
average_speed_test = final_test[total_travel_distance_column].mean() / final_test[total_travel_time_column].mean()

# Identify rows with missing total travel distance
missing_distance_rows_test = final_test[final_test[total_travel_distance_column].isnull()]

# Impute missing distances based on average speed
final_test.loc[missing_distance_rows_test.index, total_travel_distance_column] = (
    missing_distance_rows_test[total_travel_time_column] * average_speed_test
)

# Mode imputation for 'segmentsCabinCode'
final_test[cabin_code_column].fillna(mode_cabin_code, inplace=True)

# Mean imputation for 'seatsRemaining'
final_test[seats_remaining_column].fillna(mean_seats_remaining, inplace=True)

# Convert date strings to datetime objects
final_test[date_column1] = pd.to_datetime(final_test[date_column1], format='%Y-%m-%d')
final_test[date_column2] = pd.to_datetime(final_test[date_column2], format='%Y-%m-%d')

# Create a new column for the day difference
final_test['dayDifference'] = (final_test[date_column1] - final_test[date_column2]).dt.days

# Create a new column to track the number of connections
final_test['connections'] = final_test['segmentsArrivalAirportCode'].apply(lambda x: x.count('||'))

# Select numeric and categorical features
final_test_num = final_test.select_dtypes(include=np.number)
final_test_cat = final_test.select_dtypes(exclude=np.number)

# Skewness correction for numeric features
skewness_test = final_test_num.skew().sort_values(ascending=False)
skewed_vars_test = skewness_test.index[skewness_test >= 1].to_list()
unskewed_num_data_test = final_test_num.drop(skewed_vars_test, axis=1)
skewed_num_data_test = final_test_num[skewed_vars_test]
adjust_skew_test = np.log1p(skewed_num_data_test)
final_test_num = pd.concat([unskewed_num_data_test, adjust_skew_test], axis=1)

# Convert categorical features to numeric
final_test_cat[starting_airport_column] = final_test_cat[starting_airport_column].replace(airport_mapping)
final_test_cat[destination_airport_column] = final_test_cat[destination_airport_column].replace(airport_mapping)
final_test_cat[cabin_code_column] = final_test_cat[cabin_code_column].str.replace(' ','')
final_test_cat[cabin_code_column] = final_test_cat[cabin_code_column].str.replace('||',' ')
final_test_cat[cabin_code_column] = final_test_cat[cabin_code_column].str.split()
final_test_cat['cabinCodeNums'] = final_test_cat[cabin_code_column].apply(lambda x: [individual_cabin_mapping.get(code, code) for code in x])
final_test_cat["averageCabinCode"] = final_test_cat['cabinCodeNums'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
num_from_cat_vars_test = ['startingAirport','destinationAirport', 'averageCabinCode']
num_from_cat_test = final_test_cat[num_from_cat_vars_test]
num_from_cat_test = num_from_cat_test.astype(int)
final_test_cat = final_test_cat.drop(num_from_cat_test, axis=1)
final_test_cat = final_test_cat.drop(['segmentsCabinCode','cabinCodeNums'], axis=1)

# Concatenate all transformed features
final_test_allnum = pd.concat([final_test_num, num_from_cat_test, final_test_cat['id']], axis=1)
final_test_allnum = final_test_allnum.set_index("id")

train_set_allnum = train_set_allnum.set_index("id")
# separate data label from the features in the train set 

train_label = train_set_allnum["price"]  # this is log_SalePrice

train_features = train_set_allnum.drop("price", axis = 1)

test_set_allnum = test_set_allnum.set_index("id")
test_label = test_set_allnum["price"]# this is log_SalePrice

test_features = test_set_allnum.drop("price", axis = 1)

"""
# Instantiate the Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(random_state=42)

# Fit the model on the training data
gb_model.fit(train_features, train_label)

# Predict prices on the test set
gb_predictions = gb_model.predict(test_features)

# Evaluate the model
gb_mse = mean_squared_error(test_label, gb_predictions)
gb_r2 = r2_score(test_label, gb_predictions)

#r2 = 0.5578035367494885
"""
rf_model = RandomForestRegressor(random_state=42)

# Fit the model on the training data
rf_model.fit(train_features, train_label)

# Predict prices on the test set
predictions = rf_model.predict(test_features)

# Evaluate the model
mse = mean_squared_error(test_label, predictions)
r2 = r2_score(test_label, predictions)

#r2 = 0.6604299329390051

def obs_vs_pred_test(observation, prediction):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(observation, prediction)
    ax.plot([0, max(observation)], [0, max(observation)], color='red')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # ax.set_ylim(0, prediction.max())
    ax.set_xlabel("observed Sale Price")
    ax.set_ylabel("predicted Sale Price")
    ax.set_title("Test Set")
    plt.show()

obs_vs_pred_test(test_label, predictions)

finalPredictions = rf_model.predict(final_test_allnum)

final_predictions = pd.DataFrame({'id': final_test_allnum.index, 'predicted_price': finalPredictions})

# Save the DataFrame to a CSV file
final_predictions.to_csv('prediction.csv', index=False)

