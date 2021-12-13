import numpy as np

def load_data(data, num_features, num_data=-1):        
    x_list, y_list = [], []
    for line in data:
        x, motion, y = line.rstrip().split('\t')
        if num_features == 1:
            x = int(x)
        else:
            x = list(map(int, x.zfill(num_features)))
        y = int(y)
        x_list.append(x)
        y_list.append(y)
    x_list = x_list[:num_data]
    y_list = y_list[:num_data]
    if num_features == 1:
        x_array=np.array(x_list).reshape(-1, 1)
    else:
        x_array=np.array(x_list)
        
    y_array=np.array(y_list).reshape(-1, 1)

    data_array = np.concatenate((x_array, y_array), axis=1)

    return data_array

def multi_acc(pred_y, batch_y):
    pred_y = np.concatenate(pred_y)
    batch_y = np.concatenate(batch_y)
    pred_y_label = np.argmax(pred_y, axis=1)    
    correct = (pred_y_label == batch_y)

    acc = correct.sum() / len(correct)
    acc = np.round(acc * 100)
    
    return acc.item()