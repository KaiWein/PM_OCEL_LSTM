from functions import (inbu,prep,setting,folding)
from keras.models import load_model
from jellyfish import damerau_levenshtein_distance, levenshtein_distance
from sklearn import metrics
import numpy as np
import pandas as pd
import distance

setting_inputs = setting.inputdef()
csvname = f"{setting_inputs['flatten_by']}_complete" if setting_inputs['complete'] else f"{setting_inputs['flatten_by']}_filter"
# Define model_file based on single_log value
model_file = f"{csvname}_single" if setting_inputs['single_log'] else f"{csvname}_enriched"

# prep the ocel and reading
ocel, act_dict, cust_dict = prep.prep_ocel_complete(setting_inputs=setting_inputs,csvname=csvname)
ocel_train, ocel_test = folding.folding_train_test(ocel, csvname= model_file)
## define some static variables 
divisor = np.mean(ocel['Time_Diff'])  # average time between events
divisor2 = np.mean(ocel['Time_Since_Start'])  # average time between current and first events
divisorTR = np.mean(ocel['Remaining_Time'])  # average time instance remaining
divisor3 = ocel.groupby('Case_ID')['Time_Since_Start'].apply(lambda x: (x.iloc[-1] - x).mean()).mean()
num_of_features, max_trace_length, act_feat,cust_feat, target_act_feat, target_act_feat_dict, other_features = setting.feature_dimensios(ocel=ocel,setting_input=setting_inputs)

modelname = input('Enter the model name (in the directory output_files/models):')
if model_file not in modelname:
    raise ValueError(f"{model_file} is not a substring of {modelname}")

model = load_model(f'./output_files/models/{modelname}')



X_test, y_test_a, y_test_t, y_test_tr = inbu.generating_inputs(ocel_fold=ocel_test,ocel=ocel,setting_input=setting_inputs,
                                                                  dn= divisor, ds= divisor2, dr= divisorTR)

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
print(f'The accuracy of the activation prediction is {sum(act_comp)/len(act_comp)}')
MAE_Time_diff = metrics.mean_absolute_error(output_ocel['Pred_Time_Diff']/ (24 * 60 * 60),output_ocel['Next_Time_Diff']/ (24 * 60 * 60))
MAE_rem_time = metrics.mean_absolute_error(output_ocel['Pred_Remaining_Time']/ (24 * 60 * 60),output_ocel['Next_Remaining_Time']/ (24 * 60 * 60))
RMSE_Time_diff = metrics.mean_squared_error(output_ocel['Pred_Time_Diff']/ (24 * 60 * 60),output_ocel['Next_Time_Diff']/ (24 * 60 * 60),squared=False)
RMSE_rem_time = metrics.mean_squared_error(output_ocel['Pred_Remaining_Time']/ (24 * 60 * 60),output_ocel['Next_Remaining_Time']/ (24 * 60 * 60),squared=False)
print(f'MAE of the time between events in days {MAE_Time_diff}')
print(f'MAE of the remaining time in days {MAE_rem_time}')
# print(f'RMSE of the time between events in days {RMSE_Time_diff}')
# print(f'RMSE of the remaining time in days {RMSE_rem_time}')