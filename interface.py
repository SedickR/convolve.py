from argparse import ArgumentParser


def analyser_la_ligne_de_commande():
    """Crée une interface pour la ligne de commande"""
    parser = ArgumentParser(description="Analyse spot size")
    parser.add_argument('load', help="Load a directory or a file", nargs='+')
    parser.add_argument('-a', '--analyse', help='Analyse rms spot size from image', action='store_true')
    parser.add_argument('-p', '--plot', help='Plot spot size', action='store_true')
    parser.add_argument('-s', '--save',
                        help='Save data to csv format', action='store_true')
    parser.add_argument(
        '-v', '--verbose', help='Show steps in details', action='store_true')
    parser.add_argument('-rgb', help = 'Select a color channel',
                        action='store_true')
    return parser.parse_args()