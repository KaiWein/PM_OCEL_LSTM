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

# Generate inputs with varying prefix lengths
prefix_lengths = range(2, max_trace_length-1)  # List of prefix lengths to consider
results = []

for prefix_length in prefix_lengths:
    print(f"Results for Prefix Length {prefix_length}:")
    
    # Generate inputs with the current prefix length
    X_test,y_test_a, y_test_t, y_test_tr = inbu.generating_inputs(ocel_fold=ocel_test,ocel=ocel,setting_input=setting_inputs,
                                                                  dn= divisor, ds= divisor2, dr= divisorTR,
                                                                    prefix_length=prefix_length)
    # Make predictions with the model
    y = model.predict(X_test, verbose=1)
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
    trace_length = ocel_test['Trace_Len'].values
    output_ocel = ocel_test[trace_length >= prefix_length].reset_index(drop= True).drop(columns=columns_to_drop_existing).copy()
    output_ocel['Pred_Activity'] = pred_act_list
    output_ocel['Pred_Time_Diff'] = y_t1
    output_ocel['Pred_Remaining_Time'] = y_tr1

    output_ocel['Levenshtein'] = output_ocel.apply(lambda row: 1 - levenshtein_distance(row['Pred_Activity'], row['Next_Activity']), axis=1)
    output_ocel['Damerau'] = output_ocel.apply(lambda row: 1 - (damerau_levenshtein_distance(row['Pred_Activity'], row['Next_Activity']) / max(len(row['Pred_Activity']), len(row['Next_Activity']))), axis=1)
    output_ocel['Damerau'] = output_ocel['Damerau'].clip(lower=0)
    output_ocel['Jaccard'] = output_ocel.apply(lambda row: 1 - distance.jaccard(row['Pred_Activity'], row['Next_Activity']), axis=1)

    mae_time_diff = metrics.mean_absolute_error(output_ocel['Pred_Time_Diff']/ (24 * 60 * 60), output_ocel['Next_Time_Diff']/ (24 * 60 * 60)) 
    mae_remaining_time = metrics.mean_absolute_error(output_ocel['Pred_Remaining_Time']/ (24 * 60 * 60), output_ocel['Next_Remaining_Time']/ (24 * 60 * 60)) 
    rmse_time_diff = metrics.mean_squared_error(output_ocel['Pred_Time_Diff']/ (24 * 60 * 60), output_ocel['Next_Time_Diff']/ (24 * 60 * 60), squared=False) 
    rmse_remaining_time = metrics.mean_squared_error(output_ocel['Pred_Remaining_Time']/ (24 * 60 * 60), output_ocel['Next_Remaining_Time']/ (24 * 60 * 60), squared=False) 

    # Store the results for the current prefix length
    results.append({
        'Prefix Length': prefix_length,
        'length': len(y_tr),
        'MAE Time Difference': mae_time_diff,
        'MAE Remaining Time': mae_remaining_time,
        'RMSE Time Difference': rmse_time_diff,
        'RMSE Remaining Time': rmse_remaining_time
    })

    # Output additional values based on Case_ID and prefix length
    for case_id in output_ocel['Case_ID'].unique():
        case_data = output_ocel[output_ocel['Case_ID'] == case_id]
        trace_length = len(case_data)
        if prefix_length <= trace_length:
            # print(f"\nAdditional values for Prefix Length {prefix_length} and Case ID {case_id}:")
            case_data_prefix = case_data[:prefix_length]
            #print(case_data_prefix.to_string(index=False))
            
            mae_time_diff_case = metrics.mean_absolute_error(case_data_prefix['Pred_Time_Diff']/ (24 * 60 * 60), case_data_prefix['Next_Time_Diff']/ (24 * 60 * 60))
            mae_remaining_time_case = metrics.mean_absolute_error(case_data_prefix['Pred_Remaining_Time']/ (24 * 60 * 60), case_data_prefix['Next_Remaining_Time']/ (24 * 60 * 60))
            rmse_time_diff_case = metrics.mean_squared_error(case_data_prefix['Pred_Time_Diff']/ (24 * 60 * 60), case_data_prefix['Next_Time_Diff']/ (24 * 60 * 60), squared=False)
            rmse_remaining_time_case = metrics.mean_squared_error(case_data_prefix['Pred_Remaining_Time']/ (24 * 60 * 60), case_data_prefix['Next_Remaining_Time']/ (24 * 60 * 60), squared=False)
            
            # print(f"\nMetrics for Prefix Length {prefix_length} and Case ID {case_id}:")
            # print("MAE Time Difference:", mae_time_diff_case)
            # print("MAE Remaining Time:", mae_remaining_time_case)
            # print("RMSE Time Difference:", rmse_time_diff_case)
            # print("RMSE Remaining Time:", rmse_remaining_time_case)

# Output the overall results
df_results = pd.DataFrame(results)
print("\nOverall Results:")
print(df_results)
