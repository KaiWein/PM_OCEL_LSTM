from functions import prep


def inputdef():
    source = "running-example"
    # Define static variables for testing
    add_customer = 0
    normalize = True
    pos_ex = False

    # Prompt user for input
    flatten_by = input("Enter the value for flatten_by (Orders, Items, or Packages): ")
    single_log = input("Enter the value for single_log (True/False): ")

    # Error handling for invalid input values
    valid_flatten_values = ['Orders', 'Items', 'Packages']
    valid_boolean_values = ['true', 'false', '1', '0']

    if flatten_by not in valid_flatten_values:
        raise ValueError(f"Invalid input: flatten_by must be one of {valid_flatten_values}")

    if single_log.lower() not in valid_boolean_values:
        raise ValueError("Invalid input: single_log must be a boolean value (True/False)")

    # Convert input values to boolean
    single_log = single_log.lower() in ['true', '1']

    # Determine the value for complete based on flatten_by value
    if flatten_by == 'Packages':
        complete = 'True'
    else:
        complete = input("Enter the value for complete (True/False): ")

    if complete.lower() not in valid_boolean_values:
        raise ValueError("Invalid input: complete must be a boolean value (True/False)")
    complete = complete.lower() in ['true', '1']
    print("Settings:")
    print(f"flatten_by: {flatten_by}")
    print(f"single_log: {single_log}")
    print(f"complete: {complete}")
    print(f"Add customer: {add_customer == 1}")
    print(f"Normalize: {normalize}")
    print(f"Position excluded: {pos_ex}\n")
    return {
        'source': source,
        'add_customer': add_customer,
        'normalize': normalize,
        'pos_ex': pos_ex,
        'flatten_by': flatten_by,
        'single_log': single_log,
        'complete': complete
    }


def feature_dimensios(ocel, setting_input):
    flatten_by = setting_input['flatten_by']
    single_log = setting_input['single_log']
    act_feat = [feat for feat in ocel.columns if feat.startswith('Act_') and not feat.startswith('Next_Act_')]
    act_feat.remove('Act_!')
    # act_feat_dict = {index: value.replace('Act_', '') for index, value in enumerate(act_feat)}
    target_act_feat = list(filter(lambda k: k.startswith('Next_Act_') and not k.startswith('Act_'), ocel.columns))
    target_act_feat_dict = {index: value.replace('Next_Act_', '') for index, value in enumerate(target_act_feat)}
    
    time_feat = ['Time_Diff', 'Time_Since_Start', 'Time_Since_Midnight', 'Weekday'] + ['Position'] * (1 - int(setting_input['pos_ex']))
    other_features = ['Amount_Items'] * (flatten_by != 'Items') + ['In_Package'] * (flatten_by != 'Packages') + ['Amount_Orders'] * (flatten_by != 'Orders')
    cust_feat = [feat for feat in ocel.columns if 'Cust_' in feat] * (1 - int(single_log)) * setting_input['add_customer']
    feature_select = act_feat + time_feat + other_features * (1 - int(single_log)) + cust_feat * (1 - int(single_log))
    num_of_features = len(feature_select) - (len(ocel['In_Package'].unique()) == 1) if 'In_Package' in ocel.columns else len(feature_select)
    max_trace_length = prep.gen_traces_and_maxlength_of_trace(ocel)[1]
    
    print(f"Length of act_feat: {len(act_feat)}, Length of cust_feat: {len(cust_feat)}")
    
    return num_of_features, max_trace_length, act_feat, cust_feat, target_act_feat, target_act_feat_dict, other_features
