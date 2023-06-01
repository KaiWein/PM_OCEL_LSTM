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
# ocel, *_ = prep.prep_ocel_complete(setting_inputs=setting_inputs,csvname=csvname)
# ocel_train, ocel_test = folding.folding_train_test(ocel, csvname= model_file)
## define some static variables 

modelname = input('Enter the model name (in the directory output_files/models):')
if model_file not in modelname:
    raise ValueError(f"{model_file} is not a substring of {modelname}")

model = load_model(f'./output_files/models/{modelname}')
ocel = pd.read_csv(f'./output_files/folds/{model_file}_all.csv')
ocel_test = pd.read_csv(f'./output_files/folds/{model_file}_test.csv')
# ocel_train = pd.read_csv(f'./output_files/folds/{model_file}_train.csv')
divisor = np.mean(ocel['Time_Diff'])  # average time between events
divisor2 = np.mean(ocel['Time_Since_Start'])  # average time between current and first events
divisorTR = np.mean(ocel['Remaining_Time'])  # average time instance remaining
divisor3 = ocel.groupby('Case_ID')['Time_Since_Start'].apply(lambda x: (x.iloc[-1] - x).mean()).mean()
num_of_features, max_trace_length, act_feat,cust_feat, target_act_feat, target_act_feat_dict, other_features = setting.feature_dimensios(ocel=ocel,setting_input=setting_inputs)

prefix_lengths = range(2, max_trace_length)  # List of prefix lengths to consider
results = []
print(max_trace_length)
for prefix_length in prefix_lengths:    
    if setting.feature_dimensios(ocel=ocel_test,setting_input=setting_inputs)[1] == prefix_length:
        continue
    print(f"Results for Prefix Length {prefix_length}:")
    # Generate inputs with the current prefix length
    X_test, y_test_a, y_test_t, y_test_tr = inbu.generating_inputs_pref(ocel_fold=ocel_test, ocel=ocel,
                                                                  setting_input=setting_inputs,
                                                                  dn=divisor, ds=divisor2, dr=divisorTR,
                                                                  prefix_length=prefix_length)
    # Make predictions with the model
    y = model.predict(X_test, verbose=1)
    y_char, y_t, y_tr = y[0][:][:], y[1][:][:], y[2][:][:]

    max_index_list = [np.argmax(pred) for pred in y_char]
    pred_act_list = [target_act_feat_dict.get(item, item) for item in max_index_list]
    y_t = np.maximum(y_t, 0)
    y_t, y_tr = y_t * divisor, y_tr * divisorTR
    y_test_t, y_test_tr = y_test_t * divisor, y_test_tr * divisorTR  
    columns_to_drop = [col for col in ocel_test.columns if 'Act_' in col] + \
                    [col for col in ocel_test.columns if 'Cust_' in col] + \
                    ['Customers', 'Next_Time_Since_Start',
                    'Next_Time_Since_Midnight', 'Next_Weekday',
                    'Position', 'Time_Since_Midnight', 'Weekday'] + other_features
    idx = (ocel_test['Trace_Len'].values >= prefix_length) *(ocel_test['Position'].values >= prefix_length)
    next_act = ocel_test.loc[idx,'Next_Activity'] 
    act_comp = pred_act_list == next_act
    mae_time_diff = metrics.mean_absolute_error(y_t / (24 * 60 * 60), y_test_t / (24 * 60 * 60))
    mae_remaining_time = metrics.mean_absolute_error(y_tr / (24 * 60 * 60), y_test_tr / (24 * 60 * 60))
    rmse_time_diff = metrics.mean_squared_error(y_t / (24 * 60 * 60), y_test_t / (24 * 60 * 60), squared=False)
    rmse_remaining_time = metrics.mean_squared_error(y_tr / (24 * 60 * 60), y_test_tr / (24 * 60 * 60), squared=False)
    # Store the results for the current prefix length
    results.append({
        'Prefix Length': prefix_length,
        'length': len(y_tr),
        'Accuracy Activity': sum(act_comp)/len(act_comp),
        'MAE Time Difference': mae_time_diff,
        'MAE Remaining Time': mae_remaining_time,
        'RMSE Time Difference': rmse_time_diff,
        'RMSE Remaining Time': rmse_remaining_time
    })

# Output the overall results
df_results = pd.DataFrame(results)
print("\nOverall Results:")
pd.display(df_results)
pd.display(df_results.mean())

