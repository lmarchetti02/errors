import logging, os
from typing import Optional, List, Tuple
from numpy import array, sqrt, ndarray, float64
from sympy import symbols, diff, lambdify
from sympy.parsing.mathematica import parse_mathematica


# global variables
logger: logging.Logger = None
logger_f: logging.Logger = None


class Functions:
    @staticmethod
    def activate_logging(log_file: Optional[str] = "log.log", **kwargs) -> None:
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
    def def_variabili(nomi: tuple) -> list:
        """
        Questa funzione definisce le variabili Sympy che vengono poi usate per i calcoli simbolici.
        Restituisce una lista con le variabili definite.

        Parametri
        ----------
        nomi: tuple
            Nomi che si vogliono assegnare alle variabili.

        Esempio
        ----------
        >>> Funzioni.def_variabili(("x", "y", "z")])
        >>> print(variabili)
        [x, y, z]
        """

        logger_f.info("Chiamata la funzione 'def_variabili()'.")

        variabili: list = [symbols(f"{nomi[i]}") for i in range(len(nomi))]
        logger_f.debug(f"Create le variabili --> {tuple(variabili)}")

        return variabili

    @staticmethod
    def derivate(G, variabili: list, misure: ndarray) -> list:
        """
        Questa funzione calcola le derivate parziali di G e le valuta in corrispondenza delle
        misure dirette (x_10,...,x_n0). In particolare modifica la lista globale `derivate` aggiungendole
        i valori appena calcolati.

        Parametri
        ----------
        G:
            Espressione Sympy della funzione G(x_1,...,x_n).
        variabili: list
            Lista delle variabili sympy x_1,x_2,...,x_n.
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

        derivate: list = []

        # calcolo espressione derivate parziali
        if len(variabili) == len(misure):
            for i in range(len(variabili)):
                derivata_simbolica = lambdify(variabili, x := diff(G, variabili[i]))
                logger_f.debug(f"Ottenuta la derivata --> {Functions.display(str(x))}.")

                derivate.append(val := derivata_simbolica(*list(misure)))
                logger_f.debug(f"Ottenuta il valore della derivata --> {val}.")
        else:
            logger_f.error(
                "Il numero di variabili non corrisponde al numero di misure!"
            )

        return array(derivate)

    @staticmethod
    def display(text: str) -> str:
        text = text.replace("**", "^")
        text = text.replace("*", "")
        text = text.replace("exp", "e^")

        return text


def propagazione_errori(
    nomi_var: tuple,
    G: str,
    x_val: ndarray,
    x_err: ndarray,
    output: bool,
    log: tuple = (True, "log.log"),
) -> float64:
    """
    Questa funzione calcola finalmente l'errore sulla grandezza G a partire dai valori ottenuti
    mediante le funzioni definite nella classe `Funzioni`.

    Parametri
    ----------
    nomi_var: tuple
        Nomi che si vogliono assegnare alle variabili.
    G: str
        Espressione della funzione G in linguaggio Mathematica.
    x_val: ndarray
        Lista Numpy con i valori [x_10,...,x_n0].
    x_err: ndarray
        Lista Numpy con i valori degli errori su [x_10,...,x_n0].
    output: bool
        Se True, viene mostrato il processo di calcolo dell'errore.
    log: tuple
        Attiva e disattiva il log della libreria, e specifica il file
        in cui si vuole salvare il log. Di default è impostata su (True, "log.log).

    Returns
    ----------
    out: float64
        Valore numerico dell'errore ottenuto dopo la propagazione.


    Esempio
    ----------
    >>> data = array([1, 2])
    >>> err = array([0.1, 0.3])
    >>> propagazione_errori(2, ("a", "b"), "a^2 + b", data, err, True)
    Variabili: [a, b]
    Funzione: a**2 + b
    Errore propagato: 0.36055512754639896
    """

    Functions.activate_logging(log[1], status=log[0])

    # definizione variabili
    variabili = Functions.def_variabili(nomi_var)

    # definizione funzione
    G = parse_mathematica(G)
    derivate = Functions.derivate(G, variabili, x_val)

    # valore funzione
    G_val = lambdify(variabili, G)(*list(x_val))
    logger.debug("Calcolato il valore di G.")

    # calcolo errore
    G_err = sqrt(((derivate * x_err) ** 2).sum())
    logger.debug("Calcolato l'errore su G.")

    if output:
        print(
            f"\nFunzione: G({','.join(list(nomi_var))}) = {Functions.display(str(G))}\n"
        )
        print(
            f"Valore misura: G({', '.join(f'{x_val[i]}' for i in range(len(nomi_var)))}) = {G_val:e}\n"
        )
        print(f"Errore propagato: {G_err:.1e}\n")
    else:
        pass

    return G_err


if __name__ == "__main__":
    propagazione_errori(
        ("x", "y", "z"),
        "x^2 + y^2 + z^2",
        array([0.1, 0.2, 0.3]),
        array([0.01, 0.05, 0.1]),
        True,
    )
