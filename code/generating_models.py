from functions import (prep, folding, inbu, LSTM_model,setting)

from matplotlib import pyplot as plt

import numpy as np
import time

np.random.seed(42)

setting_inputs = setting.inputdef()
csvname = f"{setting_inputs['flatten_by']}_complete" if setting_inputs['complete'] else f"{setting_inputs['flatten_by']}_filter"
# Define model_file based on single_log value
model_file = f"{csvname}_single" if setting_inputs['single_log'] else f"{csvname}_enriched"
time.sleep(5)
# prep the ocel and reading
ocel, act_dict, cust_dict = prep.prep_ocel_complete(setting_inputs=setting_inputs,csvname=csvname)
ocel_train, ocel_test = folding.folding_train_test(ocel, csvname= model_file)
## define some static variables 
divisor = np.mean(ocel['Time_Diff'])  # average time between events
divisor2 = np.mean(ocel['Time_Since_Start'])  # average time between current and first events
divisorTR = np.mean(ocel['Remaining_Time'])  # average time instance remaining
divisor3 = ocel.groupby('Case_ID')['Time_Since_Start'].apply(lambda x: (x.iloc[-1] - x).mean()).mean()

print(f"\ndivisor: {divisor}")
print(f"divisor2: {divisor2}")
print(f"divisorTR: {divisorTR}")
print(f"divisor3: {divisor3}")
print(f'Amount of rows of the OCEL: {len(ocel)}\n')

num_of_features, max_trace_length, act_feat,cust_feat, target_act_feat, target_act_feat_dict, other_features = setting.feature_dimensios(ocel=ocel,setting_input=setting_inputs)
number_of_train_cases = len(ocel_train)
target_act_length = len(target_act_feat)
print(f"Number of train cases: {number_of_train_cases}, Max trace length: {max_trace_length}, Number of features: {num_of_features}\n")


X_train,y_train_a, y_train_t, y_train_tr = inbu.generating_inputs(ocel_fold=ocel_train,ocel=ocel,setting_input=setting_inputs,
                                                                  dn= divisor, ds= divisor2, dr= divisorTR)

print(f"Shape of X_train: {X_train.shape}")
print(f"This matches the desired shape (number_of_train_cases, max_trace_length, num_of_features): {(number_of_train_cases, max_trace_length, num_of_features)} => {X_train.shape ==(number_of_train_cases, max_trace_length, num_of_features)}")
print(f"Shape of y_train_a: {y_train_a.shape}, this matches the desired shape (number_of_train_cases, target_act_length): {(number_of_train_cases, target_act_length)} => {y_train_a.shape ==(number_of_train_cases, target_act_length)}")
print(f"Shape of y_train_t: {y_train_t.shape}, this matches the desired shape (number_of_train_cases, ): {(number_of_train_cases, )} => {y_train_t.shape ==(number_of_train_cases, )}")
print(f"Shape of y_train_tr: {y_train_tr.shape}, this matches the desired shape (number_of_train_cases, ): {(number_of_train_cases, )} => {y_train_tr.shape ==(number_of_train_cases, )}\n")

print(f'For the following setting a model is now trained {model_file}\n')
history, best_model_name, early_stopping= LSTM_model.LSTM_MODEL(X_train, y_train_a, y_train_t, y_train_tr,filename=model_file)

val_loss = min(history.history['val_loss'])
val_loss2 = best_model_name.best
epoch = early_stopping.stopped_epoch - 49
print(f'The best value for the validation loss is  {val_loss} and was archived at the epoch {epoch}\n')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

modelname = 'model_' + model_file + f"_{epoch:02d}-{val_loss:.2f}.h5"

print(f'The best model has the name {modelname}\n')