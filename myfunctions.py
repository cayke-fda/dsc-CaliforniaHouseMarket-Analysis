#Functions cell

def root_mean_squared_error(y_true, y_pred):
    import pandas as pd
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def print_full_rows(x):
    import pandas as pd
    pd.set_option('display.max_rows', len(x))
    display(x)
    pd.reset_option('display.max_rows')

def print_full_cols(x):
    import pandas as pd
    pd.set_option('display.max_columns', len(x.columns))
    display(x)
    pd.reset_option('display.max_columns')

def print_full(x):
    import pandas as pd
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', len(x.columns))
    display(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

def type_fix(df):
    import pandas as pd
    col='Type'
    df[col].replace({'Single Family':'SingleFamily'},inplace=True)
    for key,val in df[col].value_counts().items():
        if val<100:
            df[col].replace({key:'Other'},inplace=True)
    return 0

def heating_fix(df):
    import pandas as pd
    col = 'Heating'
    df[col] = df[col].str.title()
    for key,value in df[col].value_counts().items():
        if 'Gas' in key or 'Propane' in key:
            df[col].replace({key:'Gas'},inplace=True)
        elif 'Electric' in key or 'Elec' in key:
            df[col].replace({key:'Electric'},inplace=True)
        elif 'Radiant' in key:
            df[col].replace({key:'Radiant'},inplace=True)
        elif 'Furnace' in key:
            df[col].replace({key:'Furnace'},inplace=True)
        elif 'Baseboard' in key:
            df[col].replace({key:'Baseboard'},inplace=True)
        elif 'Fireplace' in key:
            df[col].replace({key:'Fireplace'},inplace=True)
        elif 'Solar' in key:
            df[col].replace({key:'Solar'},inplace=True)
        elif 'Central' in key:
            df[col].replace({key:'Central'},inplace=True)
        elif 'None' in key:
            df[col].replace({key:'None'},inplace=True)
        elif 'Heat Pump' in key:
            df[col].replace({key:'Heat Pump'},inplace=True)
        elif 'Forced Air' in key:
            df[col].replace({key:'Forced Air'},inplace=True)
        else:
            df[col].replace({key:'Other'}, inplace=True)
    return 0

def cooling_fix(df):
    import pandas as pd
    col = 'Cooling'
    df[col] = df[col].str.title()
    for key,value in df[col].value_counts().items():
        if 'Central' in key or 'Air Conditioning' in key:
            df[col].replace({key:'Central AC'},inplace=True)
        elif 'Ceiling Fan' in key:
            df[col].replace({key:'Ceiling Fan'},inplace=True)
        elif 'Unit' in key or 'Wall' in key:
            df[col].replace({key:'Wall/Window Unit'},inplace=True)
        elif 'Evaporative' in key:
            df[col].replace({key:'Evaporative'},inplace=True)
        else:
            df[col].replace({key:'Other'},inplace=True)
    return 0

def parking_fix(df):
    import pandas as pd
    col = 'Parking'
    df[col] = df[col].str.title()
    for key,value in df[col].value_counts().items():
        if 'None' in key or '0' in key or 'On Street' in key or 'On-Street' in key or 'No Garage' in key:
            df[col].replace({key:'None'},inplace=True)
        elif 'Covered' in key:
            df[col].replace({key:'Covered Garage'},inplace=True)
        elif 'Attached' in key:
            df[col].replace({key:'Attached Garage'},inplace=True)
        elif 'Detached' in key:
            df[col].replace({key:'Detached Garage'},inplace=True)
        elif 'Carport' in key:
            df[col].replace({key:'Carport Garage'},inplace=True)
        elif 'Driveway' in key:
            df[col].replace({key:'Driveway Garage'},inplace=True)
        elif 'Off Street' in key or 'Off-Street' in key:
            df[col].replace({key:'Off Street Garage'},inplace=True)
        elif 'Assigned' in key:
            df[col].replace({key:'Assigned Garage'},inplace=True)             
        elif 'On Site' in key:
            df[col].replace({key:'On Site Garage'},inplace=True)  
        elif 'Two Door' in key or 'One Door' in key or 'Single' in key or 'Three' in key:
            df[col].replace({key:'One, Two or Three Door Garage'},inplace=True) 
        else:
            df[col].replace({key:'Other'},inplace=True) 
            
    return 0

def bedroom_fix(df):
    import pandas as pd
    col = 'Bedrooms'
    for key,value in df[col].value_counts().items():
        if key.isnumeric():
            df[col].replace({key:int(key)},inplace=True)
        else:
            count = 0
            count += key.count('Suite') +key.count('Bedroom')
            df[col].replace({key:count},inplace=True)
    return 0

def region_fix(df):
    import pandas as pd
    col = 'Region'
    th = 100 #If there aren't 100 houses sold in the region, we aggregate them as Other
    for key,value in df[col].value_counts().items():
        if value<th:
            df[col].replace({key:'Other'},inplace=True)

def top_of_the_feat(df,col):
    #This returns a list with the most common categories in a feature (over 100 residencies with it)
    import re
    import pandas as pd
    import numpy as np
    
    delimiters = ', ', ' / ', '/', '&','-'
    regex_pattern = '|'.join(map(re.escape, delimiters))
    top_floorings = ['other']
    floor_dict = {}
    for key, value in df[col].value_counts().items():
        
        new_list = re.split(regex_pattern,key.title())
        for elem in new_list:
            # print(elem)
            elem = elem.lower().replace(" ", "")
            # print(elem)
            if elem not in floor_dict.keys():
                floor_dict[elem]=0
            else:
                floor_dict[elem]+=1

    for key, value in pd.Series(floor_dict).sort_values(ascending=False).items():
        if value<100:
            break
        top_floorings.append(key.lower().replace(" ", ""))
    return top_floorings

def top_of_the_feat_encoder(df,col,top_feats):
    import pandas as pd    
    import re
    import numpy as np    
    delimiters = ', ', ' / ', '/', '&','-'
    regex_pattern = '|'.join(map(re.escape, delimiters))
    
    for key, value in df[col].value_counts().items():
        new_list = []
        split_floorings = re.split(regex_pattern,key)
        for i in range(len(split_floorings)):
            # print(split_floorings[i])
            elem = split_floorings[i].lower().replace(" ", "")
            # print(elem)
            # if 'Garbage' in split_floorings[i] :
                # print(split_floorings[i])
                # print(split_floorings[i].replace(" ", ""))
            if elem not in top_feats:
                # if 'Garbage' in split_floorings[i] :
                    # print(split_floorings[i],1)
                if new_list.count('Other')<1:
                    new_list.append('Other')
            else:
                # if 'Garbage' in split_floorings[i] :
                    # print(split_floorings[i],2)
                if elem not in new_list:
                    new_list.append(elem)
        df[col].replace({key:', '.join(sorted(new_list))},inplace=True)

    column_names = [col + '_'+ u.lower().replace(" ", "") for u in top_feats]
    df_encoded = pd.DataFrame(data=np.zeros((len(df),len(top_feats))),columns=column_names)
    
    i=0
    for entry in df[col]:
        if type(entry) != type('a'):
            continue
        entry = re.split(regex_pattern, entry)
        for elem in entry:
            df_encoded[col + '_'+ elem.lower().replace(" ", "")].iloc[i] = 1
        i+=1
    df_encoded.drop(col + '_' + top_feats[0].lower().replace(" ",''),axis=1,inplace=True)
    df.drop(col,axis=1,inplace=True)
    df = pd.concat([df,df_encoded],axis=1)
    
    return df

def listedon_fix(df):
    import pandas as pd
    import numpy as np
    col = 'Listed On'
    df[col] = pd.to_datetime(df[col])
    df[col] = (df[col] - df[col].max()).astype(str)
    for i in range(len(df[col])):
        df[col].iloc[i] = np.abs(int(df[col].iloc[i][:-5]))
    df[col].astype(float)

def state_fix(df):
    import pandas as pd
    col = 'State'
    df.replace({'CA':1,'AZ':0}, inplace=True)
    df.rename({'State':'California'}, inplace=True)

def delete_columns(df):
    import pandas as pd
    columns = ['Id','Address','Summary','Elementary School', 'Middle School', 
               'High School', 'Heating features', 'Cooling features', 'Parking features', 
               'City', 'Zip','Last Sold On']
    df.drop(columns, axis=1, inplace=True)
