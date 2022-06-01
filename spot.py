from traitement import Simulation
from rms_spot import image_analysis
from argparse import ArgumentParser
import pathlib


def analyser_la_ligne_de_commande():
    """Crée une interface pour la ligne de commande"""
    parser = ArgumentParser(description="Analyse spot size")
    parser.add_argument("load", help="Load a directory or a file", nargs="+")
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
        "-f",
        "--full",
        help="Compute full array of elemental image (diagonal, horizontal, vertical)",
        action="store_true",
    )
    parser.add_argument("-rgb", help="Select a color channel", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    terminal = analyser_la_ligne_de_commande()

    if terminal.verbose:
        print("Command line arguments:")
        print(terminal)

    if terminal.save:
        save = True
    else:
        save = False

    if terminal.rgb:
        # ask user for rgb channel specification
        rgb = input("Which color channel do you want to analyse? (r, g, b or all) ")
    else:
        rgb = "r"

    if terminal.analyse:
        analyse = image_analysis(terminal.load[0])
        results = analyse.compute_rms(channel=rgb, show=terminal.verbose)
        # ask to save if not already done
        if save is False:
            choice = input("Do you want to save the results? (y/n) ")
            if choice == "y":
                save = True
            else:
                save = False

    if terminal.full:
        analyse = image_analysis(terminal.load[0])
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
        filename = input("Enter a filename: ") + '.csv'
        analyse.save_to_csv(results, filename)

    if terminal.plot:
        choice = input("Do you want to save plots? (y/n) ")
        if choice == "y":
            save = True
        else:
            save = False
        plotting = Simulation()
        plotting.load_data(terminal.load[0])
        plotting.plot_diagonal(rgb=rgb, save=save)
        plotting.plot_matrix(rgb=rgb, save=save)
