import matplotlib
import seaborn as sb
import pandas as pd  
from statsmodels.tsa.arima.model import ARIMA

matplotlib.rcParams['figure.figsize'] = (8, 4)
sb.set(font_scale=1.)


def walk_forward_loop(train_df, val_df, column_name, order=(0, 0, 0)):
    history = train_df[column_name].copy() # trening skup koji prosirujemo stvarnom vrednosti
    wf_pred = pd.Series() # serija predikcija koju iterativno popunjavamo
    
    for i in range(len(val_df)):
        wf_model = ARIMA(history, order=order).fit()
        # sacuvaj predikciju
        y_pred = wf_model.forecast(steps=1)
        wf_pred = pd.concat([wf_pred, y_pred])
        # sacuvaj stvarnu vrednost u trening skup
        true_value = pd.Series(data=val_df.iloc[i][column_name], index=[val_df.index[i]])
        history = pd.concat([history, true_value])
    
    return wf_pred

def handle_data(data,col_names):
    '''Handles data by converting from API fetched format to compatible format for models.'''
    part = data['forecast']['forecastday'][0]
    hour = part['hour']

    pressure = dew_point = wind_deg = cloud = precip = temp = wind_speed = humidity = temp_min = temp_max = 0
    for i in range(0,len(hour)):
        pressure += hour[i]['pressure_mb']
        dew_point += hour[i]['dewpoint_c']
        wind_deg += hour[i]['wind_degree']
        cloud += hour[i]['cloud']
        precip += hour[i]['precip_mm']
        temp += hour[i]['temp_c']
        wind_speed += hour[i]['wind_kph']
        humidity += hour[i]['humidity']
        temp_min = round(min(temp_min,hour[i]['temp_c']),2)
        temp_max = round(max(temp_max,hour[i]['temp_c']),2)

    temp = round(temp/len(hour),1)
    wind_speed = round(wind_speed/len(hour),1)
    humidity = round(humidity/len(hour),1)
    dew_point = round(dew_point/len(hour),1)
    pressure = round(pressure/len(hour),1)
    wind_deg = round(wind_deg/len(hour),1)
    cloud = round(cloud/len(hour)) 

    proper_data = pd.DataFrame(columns=col_names)
    proper_data.loc[0, 'temp'] = temp
    proper_data.loc[0, 'wind_speed'] = wind_speed
    proper_data.loc[0, 'humidity'] = humidity
    proper_data.loc[0, 'precipitation'] = precip
    proper_data.loc[0, 'pressure'] = pressure
    proper_data.loc[0, 'dew_point'] = dew_point
    proper_data.loc[0, 'wind_deg'] = wind_deg
    proper_data.loc[0, 'clouds_all'] = cloud
    proper_data.loc[0, 'temp_min'] = temp_min
    proper_data.loc[0, 'temp_max'] = temp_max
    proper_data.loc[0, 'date'] = data['location']['localtime'][:10]
    proper_data['date'] = pd.to_datetime(proper_data['date'])
    proper_data.set_index('date', inplace=True)
    return proper_data.astype(float)