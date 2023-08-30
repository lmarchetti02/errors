import logging, os
from typing import Optional, List, Tuple
from numpy import array, sqrt, ndarray
from sympy import symbols, diff, lambdify
from sympy.parsing.mathematica import parse_mathematica


# global variables
logger: logging.Logger = None
logger_f: logging.Logger = None
variabili = []  # lista delle variabili x_1,...,x_n
derivate = []  # lista dei valori delle derivate parziali nei punti (x_10,...,x_n0)


class Functions:
    @staticmethod
    def activate_logging(log_file: Optional[str] = "log.log", **kwargs):
        """
        Questa funzione attiva il logging della libreria propagazione.

        Parametri
        ---
        log_file: str
            File .log dove viene memorizzato il log. Di default è
            'log.log', ma può essere cambiato.

        Parametri opzionali
        ---
        status: bool
            Attiva il logging se True, lo disattiva se 'False'. Di
            default viene impostata su True.
        """
        global logger, logger_f

        # Cancella il log precedente
        with open(f"log/{log_file}", "w") as file:
            file.close()

        # Main logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Logger per le funzioni interne a graph.py
        logger_f = logging.getLogger(__name__ + ".Functions")
        logger_f.setLevel(logging.DEBUG)
        logger_f.propagate = False  # evita log ripetuti

        # Crea un handler
        handler = logging.FileHandler("log/log.log")
        handler.setLevel(logging.DEBUG)

        # Formatta l'handler
        format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(format)

        # Aggiunge l'handler al logger
        logger.addHandler(handler)
        logger_f.addHandler(handler)

        status = kwargs.get("status", True)

        if not status:
            try:
                os.remove("log.log")
                logging.disable(logging.ERROR)
            except FileNotFoundError as e:
                logger_f.exception(e)
        elif status:
            pass
        else:
            raise TypeError("The 'status' must be of the bool type.")

    @staticmethod
    def def_variabili(N: int, nomi: tuple):
        """
        Questa funzione definisce le variabili Sympy che vengono poi usate per i calcoli simbolici.
        In particolare modifica la lista globale `variabili` aggiungendole le variabili appena definite.

        Parametri
        ----------
        N: int
            Numero di variabili della funzione G(x_1,...,x_n).
        nomi: tuple
            Nomi che si vogliono assegnare alle variabili.

        Esempio
        ----------
        >>> Funzioni.def_variabili(3, ["x", "y", "z"])
        >>> print(variabili)
        [x, y, z]
        """

        logger_f.info("Chiamata la funzione 'def_variabili()'.")

        global variabili

        try:
            for i in range(N):
                variabili.append(var := symbols(f"{nomi[i]}"))
                logger_f.debug(f"Creata la variabile --> {var}")
        except IndexError as _:
            logger_f.error("Numero di variabili e nomi non corrispondono.")

    @staticmethod
    def derivate(G, N: int, misure: ndarray):
        """
        Questa funzione calcola le derivate parziali di G e le valuta in corrispondenza delle
        misure dirette (x_10,...,x_n0). In particolare modifica la lista globale `derivate` aggiungendole
        i valori appena calcolati.

        Parametri
        ----------
        G:
            Espressione Sympy della funzione G(x_1,...,x_n).
        N: int
            Numero di variabili della funzione G.
        misure: numpy.ndarray
            Lista Numpy con i valori [x_10,...,x_n0]


        Esempio
        ----------
        >>> G = "x^2 + y"
        >>> G = parse_mathematica(G) # trasforma la stringa in un'espressione Sympy
        >>> Funzioni.derivate(G, 2, np.array([1, 2]))
        >>> print(derivate)
        [2 1]
        """

        logger_f.info("Chiamata la funzione 'derivate()'.")

        global derivate

        # calcolo espressione derivate parziali
        for i in range(N):
            der = lambdify(variabili, x := diff(G, variabili[i]))
            logger_f.debug(f"Ottenuta la derivata --> {Functions.display(str(x))}.")

            derivate.append(val := der(*list(misure)))
            logger_f.debug(f"Ottenuta il valore della derivata --> {val}.")

        derivate = array(derivate)

    @staticmethod
    def display(text):
        text = text.replace("**", "^")
        text = text.replace("*", "")
        text = text.replace("exp", "e^")

        return text


if __name__ == "__main__":
    Functions.activate_logging()

    Functions.def_variabili(2, ("x", "y"))

    G = parse_mathematica("x^2 + y^2")

    Functions.derivate(G, 2, array([0.1, 0.2]))
