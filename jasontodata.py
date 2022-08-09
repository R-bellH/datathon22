import json

def remove_items(test_list, item):
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]
    return res
with open('ims-results-json.json') as f:
    data = json.load(f)
    #default dictionary for stn names
    stn_times = {}
    for entry in data:
        stn_times[entry['stn_num']] = {'date': [],'hmd_rlt': [],'tmp_air_dry': [], 'weather_current': [], 'weather_past': [],'cloud_low_cover': [], 'avg_temp': 0}
    for entry in data:
        stn_times[entry['stn_num']] ['date'].append(entry['time_obs'][:10])
        stn_times[entry['stn_num']] ['hmd_rlt'].append(entry['hmd_rlt'])
        stn_times[entry['stn_num']] ['tmp_air_dry'].append(entry['tmp_air_dry'])
        stn_times[entry['stn_num']] ['weather_current'].append(entry['weather_crr'])
        stn_times[entry['stn_num']] ['weather_past'].append(entry['weather_past_1'])
        stn_times[entry['stn_num']] ['cloud_low_cover'].append(entry['cld_low_cvr'])
    print(stn_times)
    for entry in stn_times:
        if -9999 in stn_times[entry]['tmp_air_dry']:
            stn_times[entry]['tmp_air_dry']=remove_items(stn_times[entry]['tmp_air_dry'],-9999)
        stn_times[entry]['avg_temp'] = sum(stn_times[entry]['tmp_air_dry'])/len(stn_times[entry]['tmp_air_dry'])
        print(stn_times[entry]['avg_temp'])