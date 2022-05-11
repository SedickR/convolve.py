import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
import datetime
import os

#todo 0.75217, pas de rotation cas idéal
#centre à centre deux pixel dans la matrice diagonale, 39.5 um

class Simulation:
    """Zemax optical simulation data analysis and plotting"""

    def __init__(self):
        self.df = None
        self.matrix_s = None
        self.matrix_r = None
        self.matrix_g = None
        self.matrix_b = None

    def load_data(self, filename):
        """Load data from a Zemax .txt file"""
        self.df = pd.read_csv(filename, sep=";")
        self.matrix_s = (
            np.resize(np.array(self.df.loc[:, "Spot Size"]), (12, 12)),
            "Combined",
        )
        self.matrix_r = (
            np.resize(np.array(self.df.loc[:, "R Spot Size"]), (12, 12)),
            "Red",
        )
        self.matrix_g = (
            np.resize(np.array(self.df.loc[:, "G Spot Size"]), (12, 12)),
            "Green",
        )
        self.matrix_b = (
            np.resize(np.array(self.df.loc[:, "B Spot Size"]), (12, 12)),
            "Blue",
        )

    def heatmap_maker(self, matrix):
        """Create figure from matrix"""
        fig = go.Figure(
            data=go.Heatmap(z=matrix[0], colorbar={"title": "Spot Size [μm]"})
        )
        fig.update_yaxes(title_text="Row index")
        fig.update_xaxes(title_text="Column index")
        fig.update_layout(title=f"Spot Size RSRH")
        return fig, matrix[1]

    def plot_matrix(self, save=False, rgb=False, r=False, g=False, b=False):
        """Plot the Spot Size matrix"""
        plots = []
        plots.append(self.heatmap_maker(self.matrix_r))
        if rgb:
            plots.append(self.heatmap_maker(self.matrix_r))
            plots.append(self.heatmap_maker(self.matrix_g))
            plots.append(self.heatmap_maker(self.matrix_b))
        if r:
            plots.append(self.heatmap_maker(self.matrix_r))
        if g:
            plots.append(self.heatmap_maker(self.matrix_g))
        if b:
            plots.append(self.heatmap_maker(self.matrix_b))
        if save:
            for fig, name in plots:
                fig.write_image(
                    f'matrix_heatmap\\{name}_heatmap_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
                )
                time.sleep(0.5)
                fig.write_image(
                    f'matrix_heatmap\\{name}_heatmap_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
                )
        else:
            for fig, name in plots:
                fig.show()

    def plot_diagonal(self, save=False):
        """Plot the gradient of the Spot Size matrix"""
        matrix = self.matrix_r[0]
        diag = matrix.diagonal()
        grad = np.gradient(diag)
        fig = go.Figure(
            data=go.Scatter(
                x=np.arange(len(grad)),
                y=grad,
                mode="lines",
                name="diagonal",
                line=dict(shape="linear", color="black"),
            )
        )
        grad_v = np.gradient(matrix[:, 0])
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(grad_v)),
                y=grad_v,
                mode="lines",
                name="vertical",
                line=dict(shape="linear", color="black", dash="dot"),
            )
        )
        grad_h = np.gradient(matrix[0, :])
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(grad_h)),
                y=grad_h,
                mode="lines",
                name="horizontal",
                line=dict(shape="linear", color="black", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(x=np.arange(len(matrix[:, 0])), y=matrix[:, 0], mode="lines")
        )
        fig.update_yaxes(title_text="Spot Size grandient")
        fig.update_xaxes(title_text="Distance from center")
        fig.update_layout(title=f"Spot Size Gradient RSRH")
        if save:
            fig.write_image(
                f'diagonal_gradient\\gradient_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
            )
            time.sleep(1)
            fig.write_image(
                f'diagonal_gradient\\gradient_{datetime.datetime.now().isoformat("_", "minutes").replace(":", "")}.pdf'
            )
        else:
            fig.show()

    def plot_all_files(
        self,
        directory,
        matrix=True,
        diagonal=True,
        save=False,
        rgb=False,
        r=False,
        g=False,
        b=False,
    ):
        """Plot all text file in the directory"""
        for file in os.listdir(directory):
            if file.endswith("E.txt"):
                self.load_data(directory + "\\" + file)
                if matrix:
                    self.plot_matrix(save=save, rgb=rgb, r=r, g=g, b=b)
                if diagonal:
                    self.plot_diagonal(save=save)


i = Simulation()
i.plot_all_files("data_batch")
