from traitement import Simulation
from rms_spot import image_analysis
from argparse import ArgumentParser
import pathlib
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter.filedialog import asksaveasfilename
import tkinter as Tk


def analyser_la_ligne_de_commande():
    """Crée une interface pour la ligne de commande"""
    parser = ArgumentParser(description="Analyse spot size")
    parser.add_argument("-f", "--folder", help="Load a folder instead of a file", action="store_true")
    parser.add_argument(
        "-a", "--analyse", help="Analyse rms spot size from image", action="store_true"
    )
    parser.add_argument("-p", "--plot", help="Plot spot size", action="store_true")
    parser.add_argument(
        "-s", "--save", help="Save data to csv format", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose", help="Show steps in details", action="store_true"
    )
    parser.add_argument(
        '-ar',
        "--array",
        help="Compute full array of elemental image (diagonal, horizontal, vertical)",
        action="store_true",
    )
    parser.add_argument("-rgb", help="Select a color channel", action="store_true")

    parser.add_argument("-l", "--line_plot", help="Compute and graph the plot size of series of images", action="store_true")
    parser.add_argument("-gen", "--generate", help="Generate an image from a zemax text file", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":



    terminal = analyser_la_ligne_de_commande()

    if terminal.folder:
        root = Tk.Tk()
        root.withdraw()
        filepath = askdirectory(title="Select a directory")
        #filepath = pathlib.Path(filepath)
    else:
        root = Tk.Tk()
        root.withdraw()
        filepath = askopenfilename(title="Open text file", filetypes=[('all files', '.*'), ('TXT', '.txt'), ('CSV', '.csv'), ('BMP', '.bmp')])
        #filepath = pathlib.Path(filepath)

    if terminal.verbose:
        print("Command line arguments:")
        print(terminal)

    if terminal.save:
        save = True
    else:
        save = False

    if terminal.rgb:
        # ask user for rgb channel specification
        rgb = input("Which color channel do you want to analyse? (r, g, b)")
    else:
        rgb = "r"

    if terminal.analyse:
        analyse = image_analysis(filepath)
        results = analyse.compute_rms(channel=rgb, show=terminal.verbose)
        # ask to save if not already done
        if save is False:
            choice = input("Do you want to save the results? (y/n) ")
            if choice == "y":
                save = True
            else:
                save = False

    if terminal.generate:
        i = Simulation()
        img = i.generate_image(filepath, rgb)
        
        analyse = image_analysis(img = img)

        results = analyse.compute_rms(channel=rgb, show=terminal.verbose)

        print(results)

        # ask to save if not already done
        """if save is False:
            choice = input("Do you want to save the results? (y/n) ")
            if choice == "y":
                save = True
            else:
                save = False"""

    if terminal.array:
        analyse = image_analysis(filepath)
        results = analyse.complete_ei(channel=rgb, show=terminal.verbose)
        # ask to save if not already done
        if save is False:
            choice = input("Do you want to save the results? (y/n) ")
            if choice == "y":
                filename = input("Enter a filename: ") + ".csv"
                analyse.save_to_csv(results, filename, full=True)
            else:
                save = False

    if save:
        # ask for file name
        filename = asksaveasfilename(title="Save results", filetypes=[('CSV', '.csv'), ('all files', '.*')])
        analyse.save_to_csv(results, filename)

    if terminal.plot:
        choice = input("Do you want to save plots? (y/n) ")
        if choice == "y":
            save = True
        else:
            save = False
        plotting = Simulation()
        plotting.load_data(filepath)
        plotting.plot_diagonal(rgb=rgb, save=save)
        #plotting.plot_matrix(rgb=rgb, save=save)

    if terminal.line_plot:
        choice = input("Do you want to save plots? (y/n) ")
        if choice == "y":
            save = True
        else:
            save = False
        plotting = Simulation()
        plotting.plot_line(filepath, rgb=rgb, save=save)


        
