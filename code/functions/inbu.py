import numpy as np
from functions import prep,setting

def generating_inputs(ocel_fold,ocel,setting_input, dn, ds,dr):
    poex = setting_input['pos_ex']
    normalize = setting_input['normalize']
    nof, mtl, act,custf, taf, *_ = setting.feature_dimensios(ocel=ocel,setting_input=setting_input)
    
    pack_flag = False
    if 'In_Package' in ocel_fold.columns:
        pack_flag =  len(ocel_fold['In_Package'].unique()) > 1
    item_flag = 'Amount_Items' in ocel_fold.columns
    order_flag = 'Amount_Orders' in ocel_fold.columns
    
    notr = len(ocel_fold)
    act_pos = len(act) 
    X = np.zeros((notr, mtl, nof), dtype=np.float32)

    if len(custf) != 0:
        cust_pos = len(custf)
        onehot_offset = act_pos + cust_pos
        cust_values = ocel_fold[custf].values
    else:
        onehot_offset = act_pos
    i = 5 - int(poex)
    if pack_flag:
        in_package_values = ocel_fold['In_Package'].values 
        pos_In_Package = onehot_offset + i
        i = i +1
    if item_flag:
        max_amount_item = ocel_fold['Amount_Items'].max()
        amount_items_values = ocel_fold['Amount_Items'].values #/ max_amount_item if normalize else  ocel_fold['Amount_Items'].values 
        pos_Amount_Items = onehot_offset + i
        i = i +1
    if order_flag:
        max_amount_order = ocel_fold['Amount_Orders'].max()
        amount_orders_values = ocel_fold['Amount_Orders'].values #/ max_amount_order if normalize else ocel_fold['Amount_Orders'].values
        pos_orders_Items = onehot_offset + i
        i = i +1
    pos_Time_Diff = onehot_offset
    pos_Time_Since_Start = onehot_offset + 1
    pos_Time_Since_Midnight = onehot_offset + 2
    pos_Weekday = onehot_offset + 3
    if not poex:
        pos_Position = onehot_offset + 4

    act_values = ocel_fold[act].values[: :] 
    position_values = ocel_fold['Position'].values 
    time_diff_values = ocel_fold['Time_Diff'].values / dn if normalize else ocel_fold['Time_Diff'].values
    time_start_values = ocel_fold['Time_Since_Start'].values  / ds if normalize else ocel_fold['Time_Since_Start'].values 
    time_midnight_values = ocel_fold['Time_Since_Midnight'].values  / 86400 if normalize else ocel_fold['Time_Since_Midnight'].values
    weekday_values = ocel_fold['Weekday'].values  / 7 if normalize else ocel_fold['Weekday'].values

    for i in range(notr):
        posi = position_values[i]
        leftpad = mtl - posi
        X[i, leftpad:, :act_pos] = act_values[i - posi + 1:i + 1, :]
        if len(custf) != 0:
            X[i, leftpad:, act_pos:act_pos + cust_pos] = cust_values[i - posi + 1:i + 1, :]
        X[i, leftpad:, pos_Time_Diff] = time_diff_values[i - posi + 1:i + 1]
        X[i, leftpad:, pos_Time_Since_Start] = time_start_values[i - posi + 1:i + 1]
        X[i, leftpad:, pos_Time_Since_Midnight] = time_midnight_values[i - posi + 1:i + 1]
        X[i, leftpad:, pos_Weekday] = weekday_values[i - posi + 1:i + 1]
        if not poex:
            X[i, leftpad:, pos_Position] = position_values[i - posi + 1:i + 1]

        if pack_flag:
            X[i, leftpad:, pos_In_Package] = in_package_values[i - posi + 1:i + 1]
        if item_flag:
            X[i, leftpad:, pos_Amount_Items] = amount_items_values[i - posi + 1:i + 1] 
        if order_flag:
            X[i, leftpad:, pos_orders_Items] = amount_orders_values[i - posi + 1:i + 1] 

    y_a = ocel_fold.loc[:, taf].to_numpy(dtype=np.float32)
    y_t = ocel_fold['Next_Time_Diff'].to_numpy(dtype=np.float32) / dn if normalize else ocel_fold['Next_Time_Diff'].to_numpy(dtype=np.float32)
    y_tr = ocel_fold['Next_Remaining_Time'].to_numpy(dtype=np.float32) / dr if normalize else ocel_fold['Next_Remaining_Time'].to_numpy(dtype=np.float32)

    return X, y_a, y_t, y_tr

def concatenate_values(group, column_names, lookback_length):
    concatenated_lists = []
    values = group[column_names].values
    for i in range(len(group)):
        sublist = values[max(0,i - lookback_length+1):i+1]
        concatenated_lists.append(sublist)
    return concatenated_lists



def generating_inputs_pref(ocel_fold,ocel,setting_input, dn, ds,dr, prefix_length=0):
    poex = setting_input['pos_ex']
    normalize = setting_input['normalize']
    nof, mtl, act,custf, taf, *_ = setting.feature_dimensios(ocel=ocel,setting_input=setting_input)
    pack_flag = False
    if 'In_Package' in ocel_fold.columns:
        pack_flag =  len(ocel_fold['In_Package'].unique()) > 1
    item_flag = 'Amount_Items' in ocel_fold.columns
    order_flag = 'Amount_Orders' in ocel_fold.columns
    if prefix_length == 0:
        prefix_length = mtl + 1

    idx_trace = (ocel_fold['Trace_Len'].values >= prefix_length) * (ocel_fold['Position'].values == prefix_length)
    
    act_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names=act, lookback_length=prefix_length).explode().tolist()
    act_values = [value for value, flag in zip(act_values, idx_trace) if flag]

    act_pos = len(act) 

    if len(custf) != 0:
        cust_pos = len(custf)
        onehot_offset = act_pos + cust_pos
        cust_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names=custf, lookback_length=prefix_length).explode().tolist()
        cust_values = [value for value, flag in zip(cust_values, idx_trace) if flag]
    else:
        onehot_offset = act_pos
    i = 5 - int(poex)
    if pack_flag:
        in_package_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='In_Package', lookback_length=prefix_length).explode().tolist()
        in_package_values = [value for value, flag in zip(in_package_values, idx_trace) if flag]
        pos_In_Package = onehot_offset + i
        i = i +1
    if item_flag:
        max_amount_item = ocel_fold['Amount_Items'].max()
        amount_items_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='Amount_Items', lookback_length=prefix_length).explode().tolist()
        amount_items_values = [value for value, flag in zip(amount_items_values, idx_trace) if flag]
        amount_items_values = amount_items_values #/ max_amount_item if normalize else  ocel_fold['Amount_Items'].values 
        pos_Amount_Items = onehot_offset + i
        i = i +1
    if order_flag:
        max_amount_order = ocel_fold['Amount_Orders'].max()
        amount_orders_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='Amount_Orders', lookback_length=prefix_length).explode().tolist()
        amount_orders_values = [value for value, flag in zip(amount_orders_values, idx_trace) if flag]
        amount_orders_values = ocel_fold['Amount_Orders'].values #/ max_amount_order if normalize else ocel_fold['Amount_Orders'].values
        pos_orders_Items = onehot_offset + i
        i = i +1
    pos_Time_Diff = onehot_offset
    pos_Time_Since_Start = onehot_offset + 1
    pos_Time_Since_Midnight = onehot_offset + 2
    pos_Weekday = onehot_offset + 3
    if not poex:
        pos_Position = onehot_offset + 4

    position_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='Position', lookback_length=prefix_length).explode().tolist()
    position_values = [value for value, flag in zip(position_values, idx_trace) if flag]

    time_diff_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='Time_Diff', lookback_length=prefix_length).explode().tolist()
    time_diff_values = [value for value, flag in zip(time_diff_values, idx_trace) if flag]
    time_diff_values = [num / dn for num in time_diff_values]  if normalize else time_diff_values
    
    time_start_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='Time_Since_Start', lookback_length=prefix_length).explode().tolist()
    time_start_values = [value for value, flag in zip(time_start_values, idx_trace) if flag]
    time_start_values = [num / ds for num in time_start_values]  if normalize else time_start_values

    time_midnight_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='Time_Since_Midnight', lookback_length=prefix_length).explode().tolist()
    time_midnight_values = [value for value, flag in zip(time_midnight_values, idx_trace) if flag]
    time_midnight_values = [num / 86400 for num in time_midnight_values] if normalize else time_midnight_values

    weekday_values = ocel_fold.groupby('Case_ID').apply(concatenate_values, column_names='Weekday', lookback_length=prefix_length).explode().tolist()
    weekday_values = [value for value, flag in zip(weekday_values, idx_trace) if flag]
    weekday_values = [num / 7 for num in weekday_values]  if normalize else weekday_values
    
    
    ocel_fold = ocel_fold[idx_trace].reset_index(drop= True)
    notr = len(ocel_fold)
    X = np.zeros((notr, mtl, nof), dtype=np.float32)
    leftpad = mtl - prefix_length
    for i in range(notr):
        X[i, leftpad:, :act_pos] = act_values[i]
        if len(custf) != 0:
            X[i, leftpad:, act_pos:act_pos + cust_pos] = cust_values[i]
        X[i, leftpad:, pos_Time_Diff] = time_diff_values[i]
        X[i, leftpad:, pos_Time_Since_Start] = time_start_values[i]
        X[i, leftpad:, pos_Time_Since_Midnight] = time_midnight_values[i]
        X[i, leftpad:, pos_Weekday] = weekday_values[i]
        if not poex:
            X[i, leftpad:, pos_Position] = position_values[i]

        if pack_flag:
            X[i, leftpad:, pos_In_Package] = in_package_values[i]
        if item_flag:
            X[i, leftpad:, pos_Amount_Items] = amount_items_values[i] 
        if order_flag:
            X[i, leftpad:, pos_orders_Items] = amount_orders_values[i] 

    y_a = ocel_fold.loc[:, taf].to_numpy(dtype=np.float32)
    y_t = ocel_fold['Next_Time_Diff'].to_numpy(dtype=np.float32) / dn if normalize else ocel_fold['Next_Time_Diff'].to_numpy(dtype=np.float32)
    y_tr = ocel_fold['Next_Remaining_Time'].to_numpy(dtype=np.float32) / dr if normalize else ocel_fold['Next_Remaining_Time'].to_numpy(dtype=np.float32)

    return X, y_a, y_t, y_tr
