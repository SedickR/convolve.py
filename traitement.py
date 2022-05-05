import numpy as np
import plotly.graph_objects as go
import scipy as sp
import pandas as pd
import time

class Simulation():
    '''Zemax optical simulation data analysis and plotting'''
    def __init__(self):
        self.df = None
        self.matrix = None
    
    def load_data(self, filename):
        '''Load data from a Zemax .txt file'''
        self.df = pd.read_csv(filename, sep=';')
        self.matrix = np.array(self.df.loc[:,'Spot Size'])
        self.matrix = np.rot90(np.resize(self.matrix, (12,12)), k=0)

    def plot_matrix(self,name, show=False):
        '''Plot the Spot Size matrix'''
        fig = go.Figure(data=go.Heatmap(z=self.matrix))
        if not show:
            fig.write_image(f'matrix_heatmap\\{name}.pdf')
            time.sleep(1)
            fig.write_image(f'matrix_heatmap\\{name}.pdf')
        else:
            fig.show()

    def plot_diagonal(self, show=False):
        '''Plot the derivative of the Spot Size matrix'''
        diag = np.flipud(self.matrix).diagonal()
        grad = np.gradient(diag)
        fig = go.Figure(data=go.Scatter(x=np.arange(len(grad)), y=grad, mode='lines'))
        if not show:
            fig.write_image(f'diagonal_gradient\\gradient.pdf')
            time.sleep(1)
            fig.write_image(f'diagonal_gradient\\gradient.pdf')
        else:
            fig.show()

i = Simulation()
i.load_data('data\\test.txt')
i.plot_matrix('test', True)
i.plot_diagonal(True)

