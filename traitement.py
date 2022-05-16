# todohighlight.include
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time, datetime, os, pathlib

# TODO:
# - 0.75217, pas de rotation cas idéal
# - centre à centre deux pixel dans la matrice diagonale, 39.5 um
# - Tester code avec un truc déjà connue et facile à faire à la main - DONE
# - intégrer dans l'affichage
# - Clarifier les intrants et extrants de chaque fonction, formats des arguments, etc.


class Simulation:
    """Zemax optical simulation data analysis and plotting"""

    def __init__(self):
        self.df = None
        self.matrix_s = None
        self.matrix_r = None
        self.matrix_g = None
        self.matrix_b = None

    def load_data(self, filename):
        """Load data from a Zemax .txt file

        Allows to load data from a single text file and stores it in a pandas dataframe
        as a global variable."""
        self.df = pd.read_csv(filename, sep=";")
        self.matrix_s = (
            np.resize(np.array(self.df.loc[:, "Spot Size"]), (12, 12)),
            "Combined",
        )
        self.matrix_r = (
            np.resize(
                np.array(self.df.loc[:, "R Spot Size"]),
                (
                    int(np.sqrt(len(self.df.loc[:, "R Spot Size"]))),
                    int(np.sqrt(len(self.df.loc[:, "R Spot Size"]))),
                ),
            ),
            "Red",
        )
        self.matrix_g = (
            np.resize(
                np.array(self.df.loc[:, "G Spot Size"]),
                (
                    int(np.sqrt(len(self.df.loc[:, "G Spot Size"]))),
                    int(np.sqrt(len(self.df.loc[:, "G Spot Size"]))),
                ),
            ),
            "Green",
        )
        self.matrix_b = (
            np.resize(
                np.array(self.df.loc[:, "B Spot Size"]),
                (
                    int(np.sqrt(len(self.df.loc[:, "B Spot Size"]))),
                    int(np.sqrt(len(self.df.loc[:, "B Spot Size"]))),
                ),
            ),
            "Blue",
        )

    def heatmap_maker(self, matrix):
        """Create figure from matrix

        Takes a matrix of spot size and returns a plotly Heatmap figure
        with degrees as x and y axis and microns as z axis"""
        fig = go.Figure(
            data=go.Heatmap(z=matrix[0], colorbar={"title": "Spot Size [μm]"})
        )
        fig.update_yaxes(title_text="Row index")
        fig.update_xaxes(title_text="Column index")
        fig.update_layout(title=f"Spot Size RSRH")
        return fig, matrix[1]

    def plot_matrix(self, rgb="r", save=False):
        """Plot the Spot Size matrix

        Takes a series of boolean arguments to specify which matrix to plot and
        whether to save the figure or not. The function does not return anything,
        but shows or saves the figure."""
        plots = []
        if "r" in rgb and "g" in rgb and "b" in rgb:
            plots.append(self.heatmap_maker(self.matrix_r))
            plots.append(self.heatmap_maker(self.matrix_g))
            plots.append(self.heatmap_maker(self.matrix_b))
        if rgb == "r":
            plots.append(self.heatmap_maker(self.matrix_r))
        if rgb == "g":
            plots.append(self.heatmap_maker(self.matrix_g))
        if rgb == "b":
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

    def plot_diagonal(self, rgb='r', save=False):
        """Plot the gradient of the Spot Size matrix
        
        Plots the gradient of the three axis of interest of spot size of the loaded data.
        Saves the figure if save is True."""
        if "r" in rgb and "g" in rgb and "b" in rgb:
            self.plot_diagonal(rgb='r', save=save)
            self.plot_diagonal(rgb='g', save=save)
            self.plot_diagonal(rgb='b', save=save)
        elif rgb == "r":
            matrix = self.matrix_r[0]
        elif rgb == "g":
            matrix = self.matrix_g[0]
        elif rgb == "b":
            matrix = self.matrix_b[0]
        else:
            raise ValueError("rgb must be 'r', 'g' or 'b' or a combination of those")
        
        # compute gradient
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
                line=dict(shape="linear", color="black", dash="dash"),
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
            go.Scatter(
                x=np.arange(len(matrix[:, 0])),
                y=matrix[:, 0],
                mode="lines",
                name="Horizontal",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(matrix[0, :])),
                y=matrix[0, :],
                mode="lines",
                name="Vertical",
            )
        )
        
        # Add axis labels
        fig.update_yaxes(title_text="Spot Size grandient")
        fig.update_xaxes(title_text="Distance from center")
        fig.update_layout(title=f"Spot Size Gradient RSRH")
        fig.update_layout(template="simple_white")

        # Save figure if specified
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
        rgb="r",
        save=False
    ):
        """Plot all text file in the directory
        
        Takes a directory and plots all the text files or csv files contained.
        Transfers the rgb argument to the class functions called"""
        directory = pathlib.Path(directory)
        for file in directory.iterdir():
            if file.suffix == '.txt' or file.suffix == '.csv':
                self.load_data(directory + "\\" + file)
                if matrix:
                    self.plot_matrix(rgb = rgb, save=save)
                if diagonal:
                    self.plot_diagonal(rgb = rgb, save=save)

