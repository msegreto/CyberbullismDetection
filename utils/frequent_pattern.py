import pandas as pd

def get_closed_itemsets(frequent_itemsets):
    closed = []
    for i, row_i in frequent_itemsets.iterrows():
        is_closed = True
        for j, row_j in frequent_itemsets.iterrows():
            if i != j and row_i['itemsets'].issubset(row_j['itemsets']) and row_i['support'] == row_j['support']:
                is_closed = False
                break
        if is_closed:
            closed.append({'itemsets': row_i['itemsets'], 'support': row_i['support']})
    return pd.DataFrame(closed)

def get_maximal_itemsets(frequent_itemsets):
    maximal = []
    for i, row_i in frequent_itemsets.iterrows():
        is_maximal = True
        for j, row_j in frequent_itemsets.iterrows():
            if i != j and row_i['itemsets'].issubset(row_j['itemsets']):
                is_maximal = False
                break
        if is_maximal:
            maximal.append({'itemsets': row_i['itemsets'], 'support': row_i['support']})
    return pd.DataFrame(maximal)