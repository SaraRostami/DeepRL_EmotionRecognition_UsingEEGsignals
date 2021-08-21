import pickle
from mat4py import loadmat
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from numpy import genfromtxt
import math
import gym
from statistics import mean,stdev
from gym import spaces
from stable_baselines3 import DQN
import EEGUtilities as eeg_util
import EEGDeepRl as eeg_drl
from sklearn.model_selection import KFold

fw_path = 'C:\\Users\\Sara Rostami.D\\Desktop\\EEG\\data\\'
data_util=eeg_util.DataUtilities(None,fw_path)
userids = [3,13,15,32,11,9,31,4,25,2,8,10,14,22,24]
number_of_timesteps =100


final_report1=''
final_report2=''
for uid in userids:
    trial_ids = np.array(range(40))

    kf = KFold(n_splits=10)
    accs_arousal = []
    accs_valence = []
    for train_idx,test_idx in kf.split(trial_ids):
        train_ids = trial_ids[train_idx]
        test_ids = trial_ids[test_idx]
        #print(f"user:{uid}\n train_id:{train_idx}\n test_id:{test_idx}")
        tp_val = 0
        tn_val= 0
        fp_val = 0
        fn_val= 0
        no_guess_val= 0
        tp_aro = 0
        tn_aro= 0
        fp_aro = 0
        fn_aro= 0
        no_guess_aro= 0
        env1 = eeg_drl.EEGDeepRLEnv(fw_path,uid,train_ids,'arousal')
        env2 = eeg_drl.EEGDeepRLEnv(fw_path,uid,train_ids,'valence')
        model1 = DQN("MlpPolicy", env1, verbose=1)
        model2 = DQN("MlpPolicy", env2, verbose=1)
        model1.learn(total_timesteps=number_of_timesteps)#predicting arousal
        model2.learn(total_timesteps=number_of_timesteps)#predicting valence
        for trial_id in test_ids:
            f_name = data_util.list_files_user_trial(uid, trial_id)[0]
            userID, trialID, valence, arousal, dominance, liking, X = \
                data_util.loadSingleDataset(f_name)
            X = X.to_numpy()
            action1 = None
            action2 = None
            real_arousal = 1 if arousal >= 4.50 else 0
            real_valence = 1 if valence >= 4.50 else 0
            action = None
            for obs_id in range(len(X)):
                obs = X[obs_id]
                action1, _states1 = model1.predict(obs, deterministic=True)
                action2, _states2 = model2.predict(obs, deterministic=True)
                if action1 == 1 and real_arousal == 1:
                    if action2 == 1 and real_valence == 1:
                        tp_val += 1
                    elif action2 == 0 and real_valence == 0:
                        tn_val += 1
                    elif action2 == 1 and real_valence == 0:
                        fp_val += 1
                    elif action2 == 0 and real_valence == 1:
                        fn_val += 1
                    tp_aro += 1
                    break
                elif action1 == 0 and real_arousal == 0:
                    if action2 == 1 and real_valence == 1:
                        tp_val += 1
                    elif action2 == 0 and real_valence == 0:
                        tn_val += 1
                    elif action2 == 1 and real_valence == 0:
                        fp_val += 1
                    elif action2 == 0 and real_valence == 1:
                        fn_val += 1
                    tn_aro += 1
                    break
                elif action1 == 1 and real_arousal == 0:
                    if action2 == 1 and real_valence == 1:
                        tp_val += 1
                    elif action2 == 0 and real_valence == 0:
                        tn_val += 1
                    elif action2 == 1 and real_valence == 0:
                        fp_val += 1
                    elif action2 == 0 and real_valence == 1:
                        fn_val += 1
                    fp_aro += 1
                    break
                elif action1 == 0 and real_arousal == 1:
                    if action2 == 1 and real_valence == 1:
                        tp_val += 1
                    elif action2 == 0 and real_valence == 0:
                        tn_val += 1
                    elif action2 == 1 and real_valence == 0:
                        fp_val += 1
                    elif action2 == 0 and real_valence == 1:
                        fn_val += 1
                    fn_aro += 1
                    break
            if action2 is None or action2 == 2:
                no_guess_val += 1
            if action1 is None or action1 == 2:
                no_guess_aro += 1

        print(f" Arousal: tp : {tp_aro}, tn : {tn_aro}, fp : {fp_aro}, fn : {fn_aro}, no_guess: {no_guess_aro}, acc : {(tp_aro + tn_aro)/(tp_aro + fp_aro + tn_aro + fn_aro + no_guess_aro)}")
        print(f" Valence: tp : {tp_val}, tn : {tn_val}, fp : {fp_val}, fn : {fn_val}, no_guess: {no_guess_val}, acc : {(tp_val + tn_val)/(tp_val + fp_val + tn_val + fn_val + no_guess_val)}")
        print("-------------------------------------------------")
        exit()
        accs_arousal.append((tp_aro + tn_aro)/(tp_aro + fp_aro + tn_aro + fn_aro + no_guess_aro))
        accs_valence.append((tp_val + tn_val)/(tp_val + fp_val + tn_val + fn_val + no_guess_val))
        #exit()

    report1 = f" user : {uid}, mean acc: {mean(accs_arousal)}, stdev: {stdev(accs_arousal)},  min acc: {min(accs_arousal)} ,max acc: {max(accs_arousal)}\n"
    report2 = f" user : {uid}, mean acc: {mean(accs_valence)}, stdev: {stdev(accs_valence)},  min acc: {min(accs_valence)} ,max acc: {max(accs_valence)}\n"
    #print(report1)
    final_report1 += report1
    final_report2 += report2
print("final report_Arousal:")
print(final_report1)
print("----------------------------------------------------------------------")
print("final report_Valence:")
print(final_report2)