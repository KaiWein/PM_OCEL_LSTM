import pickle
from functions import (prep, folding, inbu, LSTM_model)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model
from jellyfish import damerau_levenshtein_distance, levenshtein_distance
import distance
from sklearn import metrics
import os
np.random.seed(42)

source = "running-example"
## some static variables for testing 
testing_other_remaining = False
add_customer = 1
normalize = True

flatten_by = input("Enter the value for flatten_by (Orders, Items or Packages): ")
single_log = input("Enter the value for single_log (True/False): ")
if flatten_by == 'Packages':
    complete = 'True'
    add_customer = 0
else:
    complete = input("Enter the value for complete (True/False): ")
# Error handling for invalid input values
if flatten_by not in ['Orders', 'Items', 'Packages']:
    raise ValueError("Wrong Input: flatten_by must be one of ['Orders', 'Items', 'Packages']")

if single_log.lower() not in ['true', 'false', '1', '0']:
    raise ValueError("Wrong Input: single_log must be a boolean value (True/False)")

if complete.lower() not in ['true', 'false', '1', '0']:
    raise ValueError("Wrong Input: complete must be a boolean value (True/False)")

# Convert input values to boolean
single_log = single_log.lower() in ['true', '1']
complete = complete.lower() in ['true', '1'] 
if complete:
    csvname = flatten_by  + '_complete'
    fl = None
else:
    csvname = flatten_by  + '_filter'
    fl = prep.act_filter(flatten_by )

time_feat = ['Time_Diff', 'Time_Since_Start', 'Time_Since_Midnight','Weekday','Position']
other_features = [] + int(flatten_by != 'Items') * ['Amount_Items'] + int(flatten_by != 'Packages') * ['In_Package'] + int(flatten_by != 'Orders') * ['Amount_Orders']
drops_col_order = ["weight", "price", "Event_ID", 'Products']
print("Settings:")
print(f"flatten_by: {flatten_by}")
print(f"single_log: {single_log}")
print(f"complete: {complete}")

## prep the ocel and reading
ocel, act_dict, cust_dict = prep.prepare_flat_ocel(source, flatten_on= flatten_by , filter= fl)
print(act_dict)
print(cust_dict)

## create the enriched and some more preprocessing as well as saving the single and enriched versions
ocel = prep.gen_enriched_single_plus_csv(OCEL = ocel,flatted_by = flatten_by ,csvname = csvname, drops_col= drops_col_order, single = single_log)

## adding features
ocel =prep.generate_features(ocel, single= single_log)
## define some static variables 
divisor = np.mean(ocel['Time_Diff'])  # average time between events
divisor2 = np.mean(ocel['Time_Since_Start'])  # average time between current and first events
divisorTR = np.mean(ocel['Remaining_Time'])  # average time instance remaining
divisor3 = ocel.groupby('Case_ID')['Time_Since_Start'].apply(lambda x: (x.iloc[-1] - x).mean()).mean()

print(f"divisor: {divisor}")
print(f"divisor2: {divisor2}")
print(f"divisorTR: {divisorTR}")
print(f"divisor3: {divisor3}")
print(f'Amount of rows of the OCEL: {len(ocel)}')

#folding the data 
ocel_train, ocel_test = folding.folding_train_test(ocel,old_ver=False, csvsave= True)

act_feat = list(filter(lambda k: k.startswith('Act_') and not k.startswith('Next_Act_'), ocel.columns))
act_feat.remove('Act_!')
act_feat_dict = {index: value.replace('Act_', '') for index, value in enumerate(act_feat)}
target_act_feat = list(filter(lambda k: k.startswith('Next_Act_') and not k.startswith('Act_'), ocel.columns))
target_act_feat_dict = {index: value.replace('Next_Act_', '') for index, value in enumerate(target_act_feat)}

cust_feat = list(filter(lambda k: 'Cust_' in k, ocel.columns)) * (1 - int(single_log)) * add_customer

feature_select = act_feat + time_feat + other_features *(1 - int(single_log)) + cust_feat *(1 - int(single_log))
print(f"Length of act_feat: {len(act_feat)}, Length of cust_feat: {len(cust_feat)}")

## define dimensions of inputs
max_trace_length = prep.gen_traces_and_maxlength_of_trace(ocel)[1]
target_act_length = len(target_act_feat)
number_of_train_cases = len(ocel_train)
num_of_features = len(feature_select) if 'In_Package' not in ocel.columns else len(feature_select) - (len(ocel['In_Package'].unique()) == 1)
print(f"Number of train cases: {number_of_train_cases}, Max trace length: {max_trace_length}, Number of features: {num_of_features}")




X_train,y_train_a, y_train_t, y_train_tr = inbu.generating_inputs(OCEL=ocel_train,
                                                                  num_of_features=num_of_features,
                                                                  max_trace_length=max_trace_length,
                                                                  taf=target_act_feat,
                                                                  act=act_feat,
                                                                  custf=cust_feat,
                                                                  divisor_next=divisor,
                                                                  divisor_since=divisor2,
                                                                  divisor_remaining=divisorTR,
                                                                  normalize = normalize, 
                                                                  test = testing_other_remaining)

print(f"Shape of X_train: {X_train.shape}")
print(f"This matches the desired shape (number_of_train_cases, max_trace_length, num_of_features): {(number_of_train_cases, max_trace_length, num_of_features)} => {X_train.shape ==(number_of_train_cases, max_trace_length, num_of_features)}")
print(f"Shape of y_train_a: {y_train_a.shape}, this matches the desired shape (number_of_train_cases, target_act_length): {(number_of_train_cases, target_act_length)} => {y_train_a.shape ==(number_of_train_cases, target_act_length)}")
print(f"Shape of y_train_t: {y_train_t.shape}, this matches the desired shape (number_of_train_cases, ): {(number_of_train_cases, )} => {y_train_t.shape ==(number_of_train_cases, )}")
print(f"Shape of y_train_tr: {y_train_tr.shape}, this matches the desired shape (number_of_train_cases, ): {(number_of_train_cases, )} => {y_train_tr.shape ==(number_of_train_cases, )}")

if single_log:
    model_file = csvname + '_single'
else:
    model_file = csvname + '_enriched'
print(f'For the following setting a model is now trained {model_file}')
history, best_model_name, early_stopping= LSTM_model.LSTM_MODEL(X_train, y_train_a, y_train_t, y_train_tr,filename=model_file)

val_loss = min(history.history['val_loss'])
val_loss2 = best_model_name.best
epoch = early_stopping.stopped_epoch - 49
print(f'The best value for the validation loss is  {val_loss} and was archived at the epoch {epoch}')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

modelname = 'model_' + model_file + f"_{epoch:02d}-{val_loss:.2f}.h5"

model = load_model(f'./output_files/models/{modelname}')

X_test,y_test_a, y_test_t, y_test_tr = inbu.generating_inputs(OCEL=ocel_test,
                                                                  num_of_features=num_of_features,
                                                                  max_trace_length=max_trace_length,
                                                                  taf=target_act_feat,
                                                                  act=act_feat,
                                                                  custf=cust_feat,
                                                                  divisor_next=divisor,
                                                                  divisor_since=divisor2,
                                                                  divisor_remaining=divisorTR, 
                                                                  normalize = normalize, 
                                                                  test = testing_other_remaining)

# y_t = y_t * divisor3

y = model.predict(X_test,verbose=1)
y_char = y[0][:][:]
y_t = y[1][:][:]
y_tr = y[2][:][:]
max_index_list = [np.argmax(pred) for pred in y_char]
pred_act_list = [target_act_feat_dict.get(item, item) for item in max_index_list]
y_t = np.maximum(y_t, 0)
y_t1 = y_t * divisor
y_tr1 = y_tr * divisorTR

columns_to_drop = [col for col in ocel_test.columns if 'Act_' in col] + \
                  [col for col in ocel_test.columns if 'Cust_' in col] + \
                  ['Customers', 'Next_Time_Since_Start',
                   'Next_Time_Since_Midnight', 'Next_Weekday',
                   'Position', 'Time_Since_Midnight', 'Weekday'] + other_features

columns_to_drop_existing = [col for col in columns_to_drop if col in ocel_test.columns]
output_ocel = ocel_test.drop(columns=columns_to_drop_existing).copy()
output_ocel['Pred_Activity'] = pred_act_list
output_ocel['Pred_Time_Diff'] = y_t1
output_ocel['Pred_Remaining_Time'] = y_tr1

output_ocel['Levenshtein'] = output_ocel.apply(lambda row: 1 - levenshtein_distance(row['Pred_Activity'], row['Next_Activity']), axis=1)
output_ocel['Damerau'] = output_ocel.apply(lambda row: 1 - (damerau_levenshtein_distance(row['Pred_Activity'], row['Next_Activity']) / max(len(row['Pred_Activity']), len(row['Next_Activity']))), axis=1)
output_ocel['Damerau'] = output_ocel['Damerau'].clip(lower=0)
output_ocel['Jaccard'] = output_ocel.apply(lambda row: 1 - distance.jaccard(row['Pred_Activity'], row['Next_Activity']), axis=1)

act_comp = output_ocel['Pred_Activity'] == output_ocel['Next_Activity'] 
print(sum(act_comp)/len(act_comp))
print(metrics.mean_absolute_error(output_ocel['Pred_Time_Diff']/ (24 * 60 * 60),output_ocel['Next_Time_Diff']/ (24 * 60 * 60)))
print(metrics.mean_absolute_error(output_ocel['Pred_Remaining_Time']/ (24 * 60 * 60),output_ocel['Next_Remaining_Time']/ (24 * 60 * 60)))
print(metrics.mean_squared_error(output_ocel['Pred_Time_Diff']/ (24 * 60 * 60),output_ocel['Next_Time_Diff']/ (24 * 60 * 60),squared=False))
print(metrics.mean_squared_error(output_ocel['Pred_Remaining_Time']/ (24 * 60 * 60),output_ocel['Next_Remaining_Time']/ (24 * 60 * 60),squared=False))