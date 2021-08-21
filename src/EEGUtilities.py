import pickle
import pandas as pd
import os

f_name = 'C:\\Users\\Sara Rostami.D\\Desktop\\DEAP\\data_preprocessed_python\\'
fw_path = 'C:\\Users\\Sara Rostami.D\\Desktop\\EEG\\data\\'

class DataUtilities:

    def __init__(self,original_dataset_path,resulting_datasets_path):
        self.original_dataset_path=original_dataset_path
        self.write_path=resulting_datasets_path

    def convert1(self):
        """converts the original dataset into the desired format, saving it in multiple files.

                each file containing information for a single participant in a single trial.
                each line in the file representing the channel inputs for the time corresponding to the line number.

                Parameters
                ----------
                original_dataset_path : address of the original dataset
                write_path: saving address of the resulting files

        """
        f_name = self.original_dataset_path
        fw_path = self.write_path
        nLabel, nTrial, nUser, nChannel, nTime = 4, 40, 32, 40, 8064
        header = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz',
                  'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz',
                  'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
        for user in range(1, nUser + 1):
            with open(f_name + 's' + ('0' + str(user) if user < 10 else str(user)) + '.dat', 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                targets = data['labels']
                X = data['data']
                for trial in range(nTrial):
                    fw_name = f"participant_{user}_trial_{trial}_Valence_{targets[trial][0]}_Arousal_{targets[trial][1]}_Dominance_{targets[trial][2]}_Liking_{targets[trial][3]}.csv"
                    fw_X = open(fw_path + fw_name, 'w')
                    fw_X.write(','.join(header) + '\n')
                    for time in range(nTime):
                        features = []
                        for chanel in range(nChannel):
                            features.append(str(X[trial][chanel][time]))
                        if user > 22:
                            h1 = features
                            features = [h1[0], h1[1], h1[3], h1[2], h1[5], h1[4], h1[7], h1[6], h1[9], h1[8], h1[11],
                                        h1[10], h1[15], h1[12], h1[13], h1[14], h1[31], h1[30], h1[28], h1[29], h1[26],
                                        h1[27], h1[24], h1[25], h1[21], h1[22], h1[19], h1[20], h1[17], h1[16], h1[18],
                                        h1[23], h1[32], h1[33], h1[34], h1[35], h1[36], h1[38], h1[38], h1[39]]

                        line = ','.join(features)
                        line += '\n'
                        fw_X.write(line)
                    fw_X.close()

    def loadSingleDataset(self,fname: str):
        """loads a single dataset
                  Parameters
                  ----------
                  fpath: address of the single dataset
                  fname : name of the single dataset file in the format
                  "participant_m_trial_n_Valence_x_Arousal_y_Dominance_z_Liking_w"

                  Returns
                  -------
                  userID(=m),trialID(=n),valence(=x),arousal(=y),dominance(=z),liking(=w),X(=contents of the single dataset file)
        """

        userID,trialID,valence,arousal,dominance,liking=self.get_filename_Info(fname)
        X = pd.read_csv(self.write_path + fname)

        return userID, trialID, valence, arousal, dominance, liking, X

    def get_filename_Info(self,fname: str):
        lst = fname.strip('.csv').split('_')
        userID = lst[1]
        trialID = lst[3]
        valence = lst[5]
        arousal = lst[7]
        dominance = lst[9]
        liking = lst[11]
        return userID, trialID, valence, arousal, dominance, liking

    def list_all_files(self):
        return os.listdir(self.write_path)

    def list_files_user_trial(self,user_id=None,trial_id=None):
        res=[]
        files=self.list_all_files()
        for f in files:
            userID, trialID, valence, arousal, dominance, liking=self.get_filename_Info(f)
            if user_id is not None and trial_id is not None\
                    and str(userID)==str(user_id) and str(trial_id)==str(trialID):
                res.append(f)
            elif user_id is not None and trial_id is None\
                    and str(userID)==str(user_id):
                res.append(f)
            elif trial_id is not None and user_id is None\
                    and str(trial_id)==str(trialID):
                res.append(f)

        return res


eegUtil = DataUtilities(f_name, fw_path)
eegUtil.convert1()

