import numpy as np
from tabulate import tabulate
import pandas as pd
import csv
import ocel as ol
from sklearn.calibration import LabelEncoder

## preparing the cel
def prepare_ocel(fn):
    ocel = pd.read_csv('../sourcedata/%s.csv' % fn)
    ocel['ocel:timestamp']=  pd.to_datetime(ocel['ocel:timestamp'])
    col_transform = {"ocel:eid": "Event_ID", "ocel:timestamp": "Timestamp", "ocel:activity": "Activity", "ocel:type:items": "Items",
                    "ocel:type:products": "Products", "ocel:type:customers":"Customers", "ocel:type:orders":"Orders",
                    "ocel:type:packages":"Packages"}
    ocel = ocel.rename(columns=col_transform).reindex(columns=[ 'Event_ID', 'Activity','Timestamp', 'weight', 'price', 'Items','Products', 'Customers', 'Orders', 'Packages'])
    ocel['Items']=  ocel['Items'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Products']=  ocel['Products'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Customers']=  ocel['Customers'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Orders']=  ocel['Orders'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Packages']=  ocel['Packages'].fillna('')
    ocel['Packages']=  ocel['Packages'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x]).apply(lambda x: x[-1])
    ocel['Customers']=  ocel['Customers'].apply(lambda x: x[-1])
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(ocel['Activity'])
    act_dict = dict(zip(ocel['Activity'], encoded))
    encoded = encoder.fit_transform(ocel['Customers'])
    cust_dict = dict(zip(ocel['Customers'], encoded))
    ocel['Activity'] = ocel['Activity'].map(act_dict)
    ocel['Customers'] = ocel['Customers'].map(cust_dict)
    return ocel, act_dict, cust_dict
## flattening on order
def flatten(OCEL, on, printen = False):
    ocel_flat = OCEL.explode(on).reset_index(drop=True)
    if printen:
        print(f'size of normal OCEL:{OCEL.size}')
        pd.display(OCEL[OCEL['Event_ID']== 594])
        print(f'size of flattend OCEL:{ocel_flat.size}')
        pd.display(ocel_flat[ocel_flat['Event_ID']== 594])
        pd.display(ocel_flat.head())
    return ocel_flat

#generating the csv files for the imput

def gen_flatted_comp_csv(OCEL,flattedby, drops_col, printen = False):
    enriched_log = OCEL.set_index(flattedby).drop(columns=drops_col)
    enriched_log['amount_of_items'] = [len(t) for t in enriched_log['Items']]
    print(enriched_log)
    enriched_log.to_csv('../data/orders_complete_enriched.csv')
    single_log = OCEL.set_index(flattedby)[['Activity','Timestamp']]
    single_log.to_csv('../data/orders_complete_single.csv')
     

    if printen:
        pd.display(enriched_log[120:140])
        pd.display(single_log[120:140])
    
    
