import numpy as np
import pandas as pd


def preprocessData():
    print('Load data and select features')
    str_exp = '/Users/danielwu/Dropbox/Documents/CSCI5525/Project/'
    import os
    os.chdir(str_exp)

    tadpoleD1D2File = str_exp + 'TADPOLE_D1_D2.csv'

    Dtadpole = pd.read_csv(tadpoleD1D2File)

    # Create Diagnosis variable based on DXCHANGE
    idx_m = Dtadpole['PTGENDER'] == 'Male'
    Dtadpole.loc[idx_m, 'PTGENDER'] = 0
    idx_f = Dtadpole['PTGENDER'] == 'Female'
    Dtadpole.loc[idx_f, 'PTGENDER'] = 1

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

    
    Dtadpole.EXAMDATE = pd.to_datetime(Dtadpole.EXAMDATE)
    tadpole_grouped = Dtadpole.groupby("RID").apply(lambda x:(x["EXAMDATE"]-x["EXAMDATE"].min()).dt.days/365.25 + x["AGE"].min())
    tadpole_grouped.sort_index(inplace=True)
    Dtadpole["AGE_AT_EXAM"] = tadpole_grouped.values
    D2 = Dtadpole['D2'].copy()
    # Select Leaderboard subjects
    tadpoleLB1LB2File = str_exp + 'TADPOLE_LB1_LB2.csv'
    LB_Table = pd.read_csv(tadpoleLB1LB2File)
    LB = LB_Table['LB1'] + LB_Table['LB2']
    idx_lb = LB.values >= 1
    Dtadpole = Dtadpole[idx_lb]

    # Select features
    cog_tests_attributes = ["CDRSB", 'EcogPtTotal',
                            'MOCA', "MMSE", "RAVLT_immediate"]
    mri_measures = ['Hippocampus', 'WholeBrain',
                    'Entorhinal', 'MidTemp', 'Fusiform', 'ICV_bl']
    pet_measures = ["FDG", "AV45", "PIB"]
    csf_measures = ["ABETA_UPENNBIOMK9_04_19_17", "TAU_UPENNBIOMK9_04_19_17", "PTAU_UPENNBIOMK9_04_19_17"]
    risk_factors = ["APOE4", "AGE_AT_EXAM","AGE", "PTGENDER"]
    values = ['ADAS13', 'Ventricles', 'Diagnosis']

    Dtadpole = Dtadpole[['RID'] + values + mri_measures +
                        pet_measures + cog_tests_attributes + risk_factors + csf_measures].copy()

    # Force values to numeric
    h = list(Dtadpole)

    for i in range(5, len(h)):

        if Dtadpole[h[i]].dtype != 'float64':
            Dtadpole[h[i]] = pd.to_numeric(Dtadpole[h[i]], errors='coerce')

    # Sort the dataframe based on age for each subject
    urid = np.unique(Dtadpole['RID'].values)
    Dtadpole_sorted = pd.DataFrame(columns=h)
    for i in range(len(urid)):

        agei = Dtadpole.loc[Dtadpole['RID'] == urid[i], 'AGE']
        idx_sortedi = np.argsort(agei)
        D1 = Dtadpole.loc[idx_sortedi.index[idx_sortedi]]
        ld = [Dtadpole_sorted, D1]
        Dtadpole_sorted = pd.concat(ld)
    # Dtadpole_sorted = Dtadpole_sorted.drop(['AGE'], axis=1)

    # Save dataset
    Dtadpole_sorted.to_csv(
        str_exp + 'IntermediateData/Leaderboard_NeuralNetBagging.csv', index=False)

    # Make list of RIDs in D2 to be predicted
    idx_lb2 = LB_Table['LB2'] == 1
    LB2_RID = LB_Table.loc[idx_lb2, 'RID']
    SLB2 = pd.Series(np.unique(LB2_RID.values))
    SLB2.to_csv(str_exp + '/IntermediateData/ToPredict.csv', index=False)

    # SVM for TADPOLE
    print('Train SVM for Diagnosis and SVR for ADAS and Ventricles')
    # Read Data
    str_in = os.path.join(str_exp, 'IntermediateData',
                          'Leaderboard_NeuralNetBagging.csv')

    D = pd.read_csv(str_in)

    # Correct ventricle volume for ICV
    D = Dtadpole_sorted.copy()
    D['Ventricles_ICV'] = D['Ventricles'].values / D['ICV_bl'].values
    D['Hippocampus_ICV'] = D['Hippocampus'].values / D['ICV_bl'].values
    D['WholeBrain_ICV'] = D['WholeBrain'].values / D['ICV_bl'].values
    D['Entorhinal_ICV'] = D['Entorhinal'].values / D['ICV_bl'].values
    D['MidTemp_ICV'] = D['MidTemp'].values / D['ICV_bl'].values
    D['Fusiform_ICV'] = D['Fusiform'].values / D['ICV_bl'].values

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
    Dtrain = D.drop(['RID', 'Diagnosis', 'Ventricles', 'Hippocampus',
                     'WholeBrain', 'Entorhinal', 'MidTemp', 'Fusiform',"AGE"], axis=1).copy()

    Dtrainmat = Dtrain.as_matrix()
    Dtrainmat = Dtrainmat.astype(float)
    h = list(Dtrain)

    m = []
    s = []
    from fancyimpute import MICE
    X = MICE().complete(Dtrainmat)
    Dtrainmat = X
    for i in range(Dtrainmat.shape[1]):
        m.append(np.mean(Dtrainmat[:, i]))
        s.append(np.std(Dtrainmat[:, i]))
        Dtrainmat[:, i] = (Dtrainmat[:, i] - m[i]) / s[i]

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
    Y_FutureADAS13 = Y_FutureADAS13_temp[np.logical_not(
        idx_last_ADAS13)].copy()

    # Normalise ADAS
    m_FutureADAS13 = np.nanmean(Y_FutureADAS13)
    s_FutureADAS13 = np.nanstd(Y_FutureADAS13)
    Y_FutureADAS13_norm = (
        Y_FutureADAS13 - m_FutureADAS13) / s_FutureADAS13

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

    return Dtrainmat_Diagnosis, Y_FutureDiagnosis, RID, Dtrainmat, Dtrain


# class SklearnHelper(object):
#     def __init__(self, clf, seed=0, params=None):
#         params['random_state'] = seed
#         self.clf = clf(**params)

#     def train(self, x_train, y_train):
#         self.clf.fit(x_train, y_train)

#     def predict(self, x):
#         return self.clf.predict(x)

#     def fit(self, x, y):
#         return self.clf.fit(x, y)



if __name__ == "__main__":
    [Dtrainmat_Diagnosis, Y_FutureDiagnosis, RID_Diagnosis] = preprocessData()
    print(Dtrainmat_Diagnosis)
