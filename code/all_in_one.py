from functions import (prep, folding, inbu, LSTM_model, setting)
from matplotlib import pyplot as plt
import numpy as np
import time
from keras.models import load_model
from jellyfish import damerau_levenshtein_distance, levenshtein_distance
from sklearn import metrics
import distance
np.random.seed(42)

setting_inputs = setting.inputdef()
csvname = f"{setting_inputs['flatten_by']}_complete" if setting_inputs['complete'] else f"{setting_inputs['flatten_by']}_filter"
# Define model_file based on single_log value
model_file = f"{csvname}_single" if setting_inputs['single_log'] else f"{csvname}_enriched"
time.sleep(5)
# prep the ocel and reading
ocel, act_dict, cust_dict = prep.prep_ocel_complete(
    setting_inputs=setting_inputs, csvname=csvname)
ocel_train, ocel_test = folding.folding_train_test(ocel, csvname=model_file)
# define some static variables
divisor = np.mean(ocel['Time_Diff'])  # average time between events
# average time between current and first events
divisor2 = np.mean(ocel['Time_Since_Start'])
divisorTR = np.mean(ocel['Remaining_Time'])  # average time instance remaining
divisor3 = ocel.groupby('Case_ID')['Time_Since_Start'].apply(
    lambda x: (x.iloc[-1] - x).mean()).mean()

print(f"\ndivisor: {divisor}")
print(f"divisor2: {divisor2}")
print(f"divisorTR: {divisorTR}")
print(f"divisor3: {divisor3}")
print(f'Amount of rows of the OCEL: {len(ocel)}\n')

num_of_features, max_trace_length, act_feat, cust_feat, target_act_feat, target_act_feat_dict, other_features = setting.feature_dimensios(
    ocel=ocel, setting_input=setting_inputs)
number_of_train_cases = len(ocel_train)
target_act_length = len(target_act_feat)
print(
    f"Number of train cases: {number_of_train_cases}, Max trace length: {max_trace_length}, Number of features: {num_of_features}\n")


X_train, y_train_a, y_train_t, y_train_tr = inbu.generating_inputs(ocel_fold=ocel_train, ocel=ocel, setting_input=setting_inputs,
                                                                   dn=divisor, ds=divisor2, dr=divisorTR)

print(f"Shape of X_train: {X_train.shape}")
print(
    f"This matches the desired shape (number_of_train_cases, max_trace_length, num_of_features): {(number_of_train_cases, max_trace_length, num_of_features)} => {X_train.shape ==(number_of_train_cases, max_trace_length, num_of_features)}")
print(f"Shape of y_train_a: {y_train_a.shape}, this matches the desired shape (number_of_train_cases, target_act_length): {(number_of_train_cases, target_act_length)} => {y_train_a.shape ==(number_of_train_cases, target_act_length)}")
print(f"Shape of y_train_t: {y_train_t.shape}, this matches the desired shape (number_of_train_cases, ): {(number_of_train_cases, )} => {y_train_t.shape ==(number_of_train_cases, )}")
print(f"Shape of y_train_tr: {y_train_tr.shape}, this matches the desired shape (number_of_train_cases, ): {(number_of_train_cases, )} => {y_train_tr.shape ==(number_of_train_cases, )}\n")

print(f'For the following setting a model is now trained {model_file}\n')
history, best_model_name, early_stopping = LSTM_model.LSTM_MODEL(
    X_train, y_train_a, y_train_t, y_train_tr, filename=model_file)

val_loss = min(history.history['val_loss'])
val_loss2 = best_model_name.best
epoch = early_stopping.stopped_epoch - 49
print(
    f'The best value for the validation loss is  {val_loss} and was archived at the epoch {epoch}\n')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

modelname = 'model_' + model_file + f"_{epoch:02d}-{val_loss:.2f}.h5"

print(f'The best model has the name {modelname}\n')

model = load_model(f'./output_files/models/{modelname}')

X_test, y_test_a, y_test_t, y_test_tr = inbu.generating_inputs(ocel_fold=ocel_test, ocel=ocel, setting_input=setting_inputs,
                                                               dn=divisor, ds=divisor2, dr=divisorTR)

# y_t = y_t * divisor3
y = model.predict(X_test, verbose=1)
y_char = y[0][:][:]
y_t = y[1][:][:]
y_tr = y[2][:][:]
max_index_list = [np.argmax(pred) for pred in y_char]
pred_act_list = [target_act_feat_dict.get(
    item, item) for item in max_index_list]
y_t = np.maximum(y_t, 0)
y_t1 = y_t * divisor
y_tr1 = y_tr * divisorTR

columns_to_drop = [col for col in ocel_test.columns if 'Act_' in col] + \
    [col for col in ocel_test.columns if 'Cust_' in col] + \
    ['Customers', 'Next_Time_Since_Start',
     'Next_Time_Since_Midnight', 'Next_Weekday',
     'Position', 'Time_Since_Midnight', 'Weekday'] + other_features

columns_to_drop_existing = [
    col for col in columns_to_drop if col in ocel_test.columns]
output_ocel = ocel_test.drop(columns=columns_to_drop_existing).copy()
output_ocel['Pred_Activity'] = pred_act_list
output_ocel['Pred_Time_Diff'] = y_t1
output_ocel['Pred_Remaining_Time'] = y_tr1

output_ocel['Levenshtein'] = output_ocel.apply(
    lambda row: 1 - levenshtein_distance(row['Pred_Activity'], row['Next_Activity']), axis=1)
output_ocel['Damerau'] = output_ocel.apply(lambda row: 1 - (damerau_levenshtein_distance(
    row['Pred_Activity'], row['Next_Activity']) / max(len(row['Pred_Activity']), len(row['Next_Activity']))), axis=1)
output_ocel['Damerau'] = output_ocel['Damerau'].clip(lower=0)
output_ocel['Jaccard'] = output_ocel.apply(
    lambda row: 1 - distance.jaccard(row['Pred_Activity'], row['Next_Activity']), axis=1)

act_comp = output_ocel['Pred_Activity'] == output_ocel['Next_Activity']
print(
    f'The accuracy of the activation prediction is {sum(act_comp)/len(act_comp)}')
MAE_Time_diff = metrics.mean_absolute_error(
    output_ocel['Pred_Time_Diff'] / (24 * 60 * 60), output_ocel['Next_Time_Diff'] / (24 * 60 * 60))
MAE_rem_time = metrics.mean_absolute_error(output_ocel['Pred_Remaining_Time'] / (
    24 * 60 * 60), output_ocel['Next_Remaining_Time'] / (24 * 60 * 60))
RMSE_Time_diff = metrics.mean_squared_error(output_ocel['Pred_Time_Diff'] / (
    24 * 60 * 60), output_ocel['Next_Time_Diff'] / (24 * 60 * 60), squared=False)
RMSE_rem_time = metrics.mean_squared_error(output_ocel['Pred_Remaining_Time'] / (
    24 * 60 * 60), output_ocel['Next_Remaining_Time'] / (24 * 60 * 60), squared=False)
print(f'MAE of the time between events in days {MAE_Time_diff}')
print(f'MAE of the remaining time in days {MAE_rem_time}')
# print(f'RMSE of the time between events in days {RMSE_Time_diff}')
# print(f'RMSE of the remaining time in days {RMSE_rem_time}')
