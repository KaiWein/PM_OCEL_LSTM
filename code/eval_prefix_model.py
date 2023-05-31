import pickle
from keras.models import load_model
from jellyfish import damerau_levenshtein_distance, levenshtein_distance
import pandas as pd
import distance
from sklearn import metrics
from functions import inbu

import numpy as np
import pandas as pd

# Load data from the file
with open('output_files/settings.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Access the loaded data
num_of_features = loaded_data['num_of_features']
max_trace_length = loaded_data['max_trace_length']
target_act_feat = loaded_data['target_act_feat']
act_feat = loaded_data['act_feat']
cust_feat = loaded_data['cust_feat']
divisor = loaded_data['divisor']
divisor2 = loaded_data['divisor2']
divisorTR = loaded_data['divisorTR']
single_log = loaded_data['single_log']
target_act_feat_dict = loaded_data['target_act_feat_dict']
modelname = loaded_data['modelname']
normalize = loaded_data['normalize']
other_features = loaded_data['other_features']
model_file = loaded_data['model_file']
pos_ex = loaded_data['pos_ex']

# modelname = 'model_Orders_filter_single_128-1.30.h5'
model = load_model(f'./output_files/models/{modelname}')
ocel_test = pd.read_csv(f'./output_files/folds/{model_file}_test.csv')
ocel_train = pd.read_csv(f'./output_files/folds/{model_file}_train.csv')
# Generate inputs with varying prefix lengths
prefix_lengths = range(2, max_trace_length-1)  # List of prefix lengths to consider
results = []

for prefix_length in prefix_lengths:
    print(f"Results for Prefix Length {prefix_length}:")
    
    # Generate inputs with the current prefix length
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
                                                                    prefix_length=prefix_length,
                                                                    position_exclude=pos_ex)
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
