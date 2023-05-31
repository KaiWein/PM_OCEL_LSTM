import numpy as np
from functions import prep,setting

def generating_inputs(ocel_fold,ocel,setting_input, dn, ds,dr, prefix_length=0):
    poex = setting_input['pos_ex']
    normalize = setting_input['normalize']
    nof, mtl, act,custf, taf, *_ = setting.feature_dimensios(ocel=ocel,setting_input=setting_input)
    
    pack_flag = False
    if 'In_Package' in ocel_fold.columns:
        pack_flag =  len(ocel_fold['In_Package'].unique()) > 1
    item_flag = 'Amount_Items' in ocel_fold.columns
    order_flag = 'Amount_Orders' in ocel_fold.columns
    trace_length = ocel_fold['Trace_Len'].values
    ocel_fold = ocel_fold[trace_length >= prefix_length].reset_index(drop= True)
    
    notr = len(ocel_fold)
    act_pos = len(act) 

    if prefix_length != 0:
        mtl = prefix_length
    else:
        prefix_length = mtl + 1
    X = np.zeros((notr, mtl, nof), dtype=np.float32)

    if custf is not None:
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

    pos = ocel_fold['Position'].values
    trace_length = ocel_fold['Trace_Len'].values
    act_values = ocel_fold[act].values[: :] 
    position_values = ocel_fold['Position'].values 
    time_diff_values = ocel_fold['Time_Diff'].values / dn if normalize else ocel_fold['Time_Diff'].values
    time_start_values = ocel_fold['Time_Since_Start'].values  / ds if normalize else ocel_fold['Time_Since_Start'].values 
    time_midnight_values = ocel_fold['Time_Since_Midnight'].values  / 86400 if normalize else ocel_fold['Time_Since_Midnight'].values
    weekday_values = ocel_fold['Weekday'].values  / 7 if normalize else ocel_fold['Weekday'].values

    for i in range(notr):
        posi = min(pos[i], prefix_length)
        leftpad = mtl - posi

        X[i, leftpad:, :act_pos] = act_values[i - posi + 1:i + 1, :]
        if custf is not None:
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
