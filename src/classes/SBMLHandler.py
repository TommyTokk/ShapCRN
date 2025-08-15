from logging import log
import libsbml
import numpy as np
import datetime
from typing import List, Dict, Tuple, Optional, Union

# from src.utils.utils import print_log


class SBMLHandler:
    """
    Class to handle a SBML model with all the related operations
    """

    def __init__(
        self, model_path: Optional[str] = None, log_file: Optional[str] = None
    ):
        self.log_file = log_file
        self.document: Optional[libsbml.SBMLDocument] = None
        self.model: Optional[libsbml.Model] = None
        self._model_path: Optional[str] = model_path

        self.reader = libsbml.SBMLReader()

        if self.reader is None:
            raise Exception("SBML reader creation failed")

        self._log(f"{model_path}")

        if model_path:
            success = self.load_model(model_path)

            if not success:
                raise Exception("SBML document load failed")

    # === MODEL LOADING ===
    def load_model(self, model_path: str) -> bool:

        doc = self.reader.readSBMLFromFile(model_path)

        if doc.getNumErrors() > 0:
            return False

        self.document = doc
        return True

    def load_model_from_string(self, model_string: str) -> bool:

        doc = self.reader.readSBMLFromString(model_string)

        if doc.getNumErrors() > 0:
            return False

        self.document = doc
        return True

    # === SPECIES HANDLER ===

    def get_list_of_species_ids(self):
        return [s.getId() for s in self.model.getListOfSpecies()]  # pyright: ignore

    def get_list_of_species_names(self):
        return [s.getName() for s in self.model.getListOfSpecies()]  # pyright: ignore

    def knockout_species(self, target_species_id):
        pass

    # === REACTIONS HANDLER ===

    def get_list_of_reactions(self):
        return self.model.getListOfReactions()  # pyright: ignore

    def get_list_of_reaction_ids(self):
        return [r.getId() for r in self.model.getListOfReactions()]  # pyright: ignore

    def get_list_of_reaction_names(self):
        return [r.getName() for r in self.model.getListOfReactions()]  # pyright: ignore

    def get_list_of_reversible_reactions(self):
        reactions = self.get_list_of_reactions()

        res = []

        for r in reactions:
            if r.getReversible():
                res.append(r)

        return res

    # === PRIVATE ===

    def _log(self, msg_str: str):
        current_date = datetime.datetime.now()
        if self.log_file:
            with open(self.log_file, "a") as out:
                out.write(f"[{current_date}]: {msg_str}\n")
        else:
            print(f"[{current_date}]: {msg_str}")
