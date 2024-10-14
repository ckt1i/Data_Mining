from sklearn.preprocessing import RobustScaler

'''This script extracts the data from the D2 dataset and saves it in a csv file'''

def extract_data(original_path, new_path):
    origin_file = open(original_path, 'r')
    datas = []
    new_file = open(new_path, 'w')
    new_file.writelines("aveAllR,aveAllL,T_RC1,T_LC1,RCC1,LCC1,T_FHCC1,T_FHRC1,T_FHLC1,T_FHBC1,T_FHTC1,T_OR1\n")
    for line in origin_file:
        if not line.startswith('1'):
            continue
        numlist = [5,6,7,11,15,16,19,20,21,22,23,27]
        i = 0
        ori_datas = line.split(',')
        data = []
        for num in numlist:
            try:
                val = ori_datas[num : num + 4*28 : 28] 
                val = [float(v) for v in val]
                avrval = sum(val) / len(val)
                data.append(avrval)
            except:
                break       

        if len(data) > 5:
            datas.append(data)

    scaler = RobustScaler()
    datas = scaler.fit_transform(datas)

    for data in datas:
        new_file.writelines(','.join([str(d) for d in data]) + '\n')
    new_file.close()
    origin_file.close()

