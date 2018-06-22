import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


print('Load data and select features')
str_exp = '/Users/danielwu/Dropbox/Documents/CSCI5525/Project/'
import os
os.chdir(str_exp)


tadpoleD1D2File = str_exp + 'TADPOLE_D1_D2.csv'

Dtadpole = pd.read_csv(tadpoleD1D2File)

# Create Diagnosis variable based on DXCHANGE
idx_mci = Dtadpole['DXCHANGE'] == 4
Dtadpole.loc[idx_mci, 'DXCHANGE'] = 2
idx_ad = Dtadpole['DXCHANGE'] == 5
Dtadpole.loc[idx_ad, 'DXCHANGE'] = 3
idx_ad = Dtadpole['DXCHANGE'] == 6
Dtadpole.loc[idx_ad, 'DXCHANGE'] = 3
idx_cn = Dtadpole['DXCHANGE'] == 7
Dtadpole.loc[idx_cn, 'DXCHANGE'] = 1
idx_mci = Dtadpole['DXCHANGE'] == 8
Dtadpole.loc[idx_mci, 'DXCHANGE'] = 2
idx_cn = Dtadpole['DXCHANGE'] == 9
Dtadpole.loc[idx_cn, 'DXCHANGE'] = 1
Dtadpole = Dtadpole.rename(columns={'DXCHANGE': 'Diagnosis'})
h = list(Dtadpole)

# Select features
Dtadpole = Dtadpole[['RID', 'Diagnosis', 'AGE',
                     'ADAS13', 'MMSE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp',  'ICV_bl']].copy()

                     # Force values to numeric
h = list(Dtadpole)
for i in range(5, len(h)):
    print(i),
    if Dtadpole[h[i]].dtype != 'float64':
        Dtadpole[h[i]] = pd.to_numeric(Dtadpole[h[i]], errors='coerce')

# Sort the dataframe based on age for each subject
urid = np.unique(Dtadpole['RID'].values)
Dtadpole_sorted = pd.DataFrame(columns=h)
for i in range(len(urid)):
    print(i),
    agei = Dtadpole.loc[Dtadpole['RID'] == urid[i], 'AGE']
    idx_sortedi = np.argsort(agei)
    D1 = Dtadpole.loc[idx_sortedi.index[idx_sortedi]]
    ld = [Dtadpole_sorted, D1]
    Dtadpole_sorted = pd.concat(ld)
Dtadpole_sorted = Dtadpole_sorted.drop(['AGE'], axis=1)

# Save dataset
Dtadpole_sorted.to_csv(
    str_exp + 'IntermediateData/BenchmarkSVMFeaturesTADPOLE.csv', index=False)

# Make list of RIDs in D2 to be predicted
idx_d2 = D2 == 1
Dtadpole_RID = Dtadpole.loc[idx_d2, 'RID']
SD2 = pd.Series(np.unique(Dtadpole_RID.values))
SD2.to_csv(str_exp + '/IntermediateData/ToPredict_D2.csv', index=False)

# SVM for TADPOLE
print('Train SVM for Diagnosis and SVR for ADAS and Ventricles')
# Read Data
str_in = os.path.join(str_exp, 'IntermediateData',
                      'BenchmarkSVMFeaturesTADPOLE.csv')

D = pd.read_csv(str_in)

# Correct ventricle volume for ICV
D['Ventricles_ICV'] = D['Ventricles'].values / D['ICV_bl'].values


# Get Future Measurements for training prediction
Y_FutureADAS13_temp = D['ADAS13'].copy()
Y_FutureADAS13_temp[:] = np.nan
Y_FutureVentricles_ICV_temp = D['Ventricles_ICV'].copy()
Y_FutureVentricles_ICV_temp[:] = np.nan
Y_FutureDiagnosis_temp = D['Diagnosis'].copy()
Y_FutureDiagnosis_temp[:] = np.nan
RID = D['RID'].copy()
uRIDs = np.unique(RID)
for i in range(len(uRIDs)):
    idx = RID == uRIDs[i]
    idx_copy = np.copy(idx)
    idx_copy[np.where(idx)[-1][-1]] = False
    Y_FutureADAS13_temp[idx_copy] = D.loc[idx, 'ADAS13'].values[1:]
    Y_FutureVentricles_ICV_temp[idx_copy] = D.loc[idx,
                                                  'Ventricles_ICV'].values[1:]
    Y_FutureDiagnosis_temp[idx_copy] = D.loc[idx, 'Diagnosis'].values[1:]
Dtrain = D.drop(['RID', 'Diagnosis'], axis=1).copy()



idx_last_Diagnosis = np.isnan(Y_FutureDiagnosis_temp)
RID_Diagnosis = RID.copy()
Dtrainmat_Diagnosis = Dtrainmat.copy()
Dtrainmat_Diagnosis = Dtrainmat_Diagnosis[np.logical_not(
    idx_last_Diagnosis), :]
RID_Diagnosis = RID_Diagnosis[np.logical_not(idx_last_Diagnosis)]
Y_FutureDiagnosis = Y_FutureDiagnosis_temp[np.logical_not(
    idx_last_Diagnosis)].copy()

# Remove NaNs in ADAS
idx_last_ADAS13 = np.isnan(Y_FutureADAS13_temp)
RID_ADAS13 = RID.copy()
Dtrainmat_ADAS13 = Dtrainmat.copy()
Dtrainmat_ADAS13 = Dtrainmat_ADAS13[np.logical_not(idx_last_ADAS13), :]
RID_ADAS13 = RID_ADAS13[np.logical_not(idx_last_ADAS13)]
Y_FutureADAS13 = Y_FutureADAS13_temp[np.logical_not(idx_last_ADAS13)].copy()

# Normalise ADAS
m_FutureADAS13 = np.nanmean(Y_FutureADAS13)
s_FutureADAS13 = np.nanstd(Y_FutureADAS13)
Y_FutureADAS13_norm = (Y_FutureADAS13 - m_FutureADAS13) / s_FutureADAS13

# Remove NaNs in Ventricles
idx_last_Ventricles_ICV = np.isnan(Y_FutureVentricles_ICV_temp)
RID_Ventricles_ICV = RID.copy()
Dtrainmat_Ventricles_ICV = Dtrainmat.copy()
Dtrainmat_Ventricles_ICV = Dtrainmat_Ventricles_ICV[np.logical_not(
    idx_last_Ventricles_ICV), :]
RID_Ventricles_ICV = RID_Ventricles_ICV[np.logical_not(
    idx_last_Ventricles_ICV)]
Y_FutureVentricles_ICV = Y_FutureVentricles_ICV_temp[np.logical_not(
    idx_last_Ventricles_ICV)].copy()

# Normalise Ventricle values
m_FutureVentricles_ICV = np.nanmean(Y_FutureVentricles_ICV)
s_FutureVentricles_ICV = np.nanstd(Y_FutureVentricles_ICV)
Y_FutureVentricles_ICV_norm = (
    Y_FutureVentricles_ICV - m_FutureVentricles_ICV) / s_FutureVentricles_ICV


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(Dense(64, input_dim=30))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Activation('softmax'))

def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model