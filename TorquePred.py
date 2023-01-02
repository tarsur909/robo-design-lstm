class TorquePred:

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import torch
    from torch.autograd import Variable

    def __init__(self, model_path, history = 8):
        self.f_lst = ['des_dx', 'x', 'dx', 'torque']
        self.history = history
        self.model = self.torch.jit.load(model_path)

    
    def create_df(self, f):
        df = self.pd.DataFrame([line.strip().split() for line in f.readlines()])
        df = df[[0, 1, 3, 6]]
        df.columns = self.f_lst
        df['vel_error'] = df['des_dx'].astype(float) - df['dx'].astype(float)
        return df

    def create_xy(self, df):
        h = self.history
        X = self.np.array(df[['x', 'vel_error']]).astype('float')
        X = self.np.array([[X[i - j] for j in range(h - 1, -1, -1)] for i in range(h - 1, len(X))])
        y = self.np.array(df[['torque']]).astype('float')[(h - 1):]
        return (X, y)

    def predict(self, dataset):
        self.dataset = dataset
        with open(self.dataset) as f:
            df = self.create_df(f)
        X, y = self.create_xy(df)
        self.model.eval()
        X = self.Variable(self.torch.tensor(X))
        whole = self.model(X.float())
        whole = whole.detach().numpy()
        whole = whole.flatten()
        result = self.pd.DataFrame(data = {'predictions': whole, 'real': y.flatten()})
        return result


    def graph_predictions(self):
       result = self.predict(self.dataset) 
       self.sns.lineplot(data = result).set(title='Torque Prediction')
       
             
    
