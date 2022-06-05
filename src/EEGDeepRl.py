import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from numpy import genfromtxt
import math
import gym
from gym import spaces
import EEGUtilities as eeg_util
from stable_baselines3 import DQN
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,roc_auc_score


fw_path = 'C:\\Users\\Sara Rostami.D\\Desktop\\EEG\\data\\'




class EEGDeepRLEnv(gym.Env):

    def __init__(self,ds_path,user_id,trial_ids_train,dimension,input_size=40):
        super(EEGDeepRLEnv, self).__init__()
        self.teacher=Teacher(ds_path,user_id,trial_ids_train,dimension)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(input_size,))

    def reset(self):
        self.teacher.load_current_episode_dataset()
        obs=self.teacher.load_current_obs()
        return obs

    def step(self, action):
        reward=self.teacher.immediate_reward(action)
        done = self.teacher.done(action)
        obs=self.teacher.load_current_obs()
        if obs is None or done:
            self.teacher.select_new_episode(reward)
            obs=self.reset()
        info = {}
        return obs, reward, done, info

    def render(self, mode):pass

    def close(self):pass
class Teacher:
    action_pos = 1
    action_neg=0
    action_nop=2
    def __init__(self,datasets_path,user_id,trial_ids_train,dimension):
        self.data_util=eeg_util.DataUtilities(None,datasets_path)
        self.user_id_level=user_id
        self.trial_id=trial_ids_train
        self.current_trial_ind=0
        self.current_obs=0
        self.dimension = dimension
        self.well_trained_trials = list()

    def load_current_episode_dataset(self):
        f_name=self.data_util.list_files_user_trial(self.user_id_level,self.trial_id[self.current_trial_ind])[0]
        userID, trialID, self.valence, self.arousal, self.dominance, self.liking, self.X=self.data_util.loadSingleDataset(f_name)
        scaler = MinMaxScaler()
        scaler.fit(self.X)
        self.X=scaler.transform(self.X)
        self.current_obs = 0

    def load_current_obs(self):
        if self.current_obs>=len(self.X):
            obs=None
        else:
            ##.loc check it
            obs=np.array(self.X[self.current_obs])
            self.current_obs += 1
        return obs

    def select_new_episode(self,agent_cumulative_result):
           if agent_cumulative_result>0:
               if self.current_trial_ind<len(self.trial_id)-1:
                   self.current_trial_ind+=1
                   self.well_trained_trials.append(f"user_id: {self.user_id_level}, trial_id: {self.trial_id[self.current_trial_ind]}")
               else:
                   self.current_trial_ind=random.randint(0,len(self.trial_id)-1)
           """TODO: else add noise to dataset"""
           self.current_obs = 0


    def immediate_reward(self,action):
        lbl=1 if(self.arousal if self.dimension == 'arousal' else self.valence if self.dimension == 'valence' else self.dominance if self.dimension == 'dominance' else self.liking) >= 4.50 else 0

        if action==self.action_pos and lbl==1:
            reward=1
        elif action==self.action_pos and lbl==0:
            reward = -1
        elif action==self.action_neg and lbl==0:
            reward = 1
        elif action==self.action_neg and lbl==1:
            reward = -1
        elif action==self.action_nop:
            if self.current_obs==len(self.X):
                reward = -5
            else:
                reward= 0

        return reward

    def done(self,action):
        lbl=1 if(self.arousal if self.dimension == 'arousal' else self.valence if self.dimension == 'valence' else self.dominance if self.dimension == 'dominance' else self.liking) >= 4.50 else 0
        if action == self.action_pos and lbl == 1:
            done=True
        elif action == self.action_pos and lbl == 0:
            done=True
        elif action == self.action_neg and lbl == 0:
            done=True
        elif action == self.action_neg and lbl == 1:
            done=True
        elif action == self.action_nop:
            if self.current_obs == len(self.X):
                done=True
            else:
                done=False

        return done








