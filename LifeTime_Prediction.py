#Data Preprocessing Libraries
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
####################################################
#Data Preprocessing Section

Data_Frame_Raw = pd.read_csv("ml_data_v2.csv")
#Data_Frame_Raw = pd.read_csv("ml_data_trimmed.csv")
Data_Frame_Raw = Data_Frame_Raw.iloc[:,1:19]



#Take input for traning models
Input_1 = (Data_Frame_Raw["bulk_taun"])
Input_2 = (Data_Frame_Raw["bulk_taup"])
Input_3 = (Data_Frame_Raw["sigma_p"])
Input_4 = (Data_Frame_Raw["sigma_n"])
Input_5 = (Data_Frame_Raw["ega_t"])
Input_6 = (Data_Frame_Raw["egd_t"])
Input_7 = (Data_Frame_Raw["wga_t"])
Input_8 = (Data_Frame_Raw["wgd_t"])
Input_9 = (Data_Frame_Raw["peak_density_nga"])
Input_10 =(Data_Frame_Raw["peak_density_ngd"])
Input_11 =(Data_Frame_Raw["generation"])

log = np.log(Input_11)
Input_12 = log*(Input_1)
Input_22 = log*(Input_2)
Input_33 = Input_11*(Input_3)
Input_44 = Input_11*(Input_4)
Input_55 = log*(Input_5)
Input_66 = log*(Input_6)
Input_77 = log*(Input_7)
Input_88 = log*(Input_8)
Input_99 = (Input_9)/Input_11
Input_101 = Input_10/Input_11                
                
Input_x = pd.concat([Input_12, Input_22, Input_33, Input_44, Input_55, Input_66, Input_77, Input_88, Input_99, Input_101, Input_11], axis=1)
sc = StandardScaler()
Input = sc.fit_transform(Input_x)

#Scaling the output parameters with hand
y = np.array(Data_Frame_Raw["lifetime_to_predict"])
y_std = np.std(y)
y_mean = np.mean(y)
Output = (y - y_mean)/y_std

#Splitting dataframe as train and test
x_train, x_test, y_train, y_test = train_test_split(Input ,Output, test_size = 0.33, random_state = 0 )


#Machine Learning Section
#Hint for both Xgb and RandomF models: More estimators more accuracy but it takes many time for training
hyperparameters = {'n_estimators': 1500, 'learning_rate': 0.14934534870723243} # best parameters based on the Ax optimization (date: 2023.03.13)
model = XGBRegressor(**hyperparameters)
model.fit(x_train, y_train)


regressor = RandomForestRegressor(n_estimators=500, random_state =0 )
regressor.fit(x_train, y_train)

all_data = pd.read_csv("ml_data_v2.csv")


def convert_to_training_format(x_in):
    Input_1 = (x_in["bulk_taun"])
    Input_2 = (x_in["bulk_taup"])
    Input_3 = (x_in["sigma_p"])
    Input_4 = (x_in["sigma_n"])
    Input_5 = (x_in["ega_t"])
    Input_6 = (x_in["egd_t"])
    Input_7 = (x_in["wga_t"])
    Input_8 = (x_in["wgd_t"])
    Input_9 = (x_in["peak_density_nga"])
    Input_10 =(x_in["peak_density_ngd"])
    Input_11 =(x_in["generation"])
    log = np.log(x_in["generation"])
    Input_12 = log*(Input_1)
    Input_22 = log*(Input_2)
    Input_33 = Input_11*(Input_3)
    Input_44 = Input_11*(Input_4)
    Input_55 = log*(Input_5)
    Input_66 = log*(Input_6)
    Input_77 = log*(Input_7)
    Input_88 = log*(Input_8)
    Input_99 = (Input_9)/Input_11
    Input_101 = Input_10/Input_11                
    Input_x = pd.concat([Input_12, Input_22, Input_33, Input_44, Input_55, Input_66, Input_77, Input_88, Input_99, Input_101, Input_11], axis=1)
    return Input_x


all_data = pd.read_csv("ml_data_v2.csv")
id_to_show = 'R2_fit_11295'

x_row = all_data[all_data.id == id_to_show]
x_row_new = x_row.iloc[[0]]
predicted_lifetime = []

for r in x_row.generation:
        x_row_new.generation = r
        x_ = x_row_new.copy()
        x = sc.transform(convert_to_training_format(x_))
        
        lifetime_pred = regressor.predict(x)
        lifetime_pred = (lifetime_pred*y_std+y_mean)
        predicted_lifetime.append(lifetime_pred[0])
        

w = savgol_filter( predicted_lifetime, 501, 3)

plt.loglog(all_data[all_data.id == id_to_show].generation, all_data[all_data.id == id_to_show].lifetime_to_predict, 'o',
all_data[all_data.id == id_to_show].generation, w)
plt.show()
