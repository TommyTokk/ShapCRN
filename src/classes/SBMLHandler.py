from logging import log
import libsbml
import numpy as np
import itertools
from typing import List, Dict, Tuple, Optional, Union
from src.classes.species import Species
from src.classes.reaction import Reaction
from src.classes.function import Function
from src.utils.utils import print_log


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
        self._model_path: Optional[str] = None

        if model_path:
            self.load_model(model_path)

    # === MODEL LOADING ===
    def load_model(self, model_path: str) -> bool:
        try:
            reader = libsbml.SBMLReader()
            self.document = reader.readSBML(model_path)

            if self.document.getNumErrors() > 0:
                self._log_errors()
                return False

            self.model = self.document.getModel()
            self._model_path = model_path

            print_log(self.log_file, f"Successfully loaded SBML model: {model_path}")
            return True

        except Exception as e:
            print_log(self.log_file, f"Error loading SBML model: {e}")
            return False

    def create_copy(self) -> "SBMLHandler":
        if not self.document:
            raise ValueError("No model loaded")

        xml_string = libsbml.writeSBMLToString(self.document)
        new_handler = SBMLHandler(log_file=self.log_file)
        new_handler.load_from_string(xml_string)
        return new_handler

    def load_from_string(self, xml_string: str) -> bool:

        try:
            reader = libsbml.SBMLReader()
            self.document = reader.readSBMLFromString(xml_string)
            self.model = self.document.getModel()

            return True
        except Exception as e:
            print_log(self.log_file, f"Error loading from string: {e}")
            return False

    # === SPECIES MANAGEMENT ===

    def get_species_ids(self) -> list[str]:
        if not self.model:
            return []

        return [s.getId() for s in self.model.getListOfSpecies()]

    def get_species_name(self) -> list[str]:
        if not self.model:
            return []
        return [s.getName() for s in self.model.getListOfSpecies()]

    def get_species_info(self) -> Dict[str, Dict]:
        """Get detailed information about all species."""
        species_info = {}
        if not self.model:
            return species_info

        for species in self.model.getListOfSpecies():
            species_info[species.getId()] = {
                "name": species.getName() if species.getName() else species.getId(),
                "initial_concentration": species.getInitialConcentration(),
                "boundary_condition": species.getBoundaryCondition(),
                "constant": species.getConstant(),
                "compartment": species.getCompartment(),
            }
        return species_info

    def get_boundary_species(self) -> List[str]:
        """Get list of species with boundary condition = true."""
        boundary_species = []
        if not self.model:
            return boundary_species

        for species in self.model.getListOfSpecies():
            if species.getBoundaryCondition():
                boundary_species.append(species.getId())
        return boundary_species

    def knockout_species(self, target_species_id: str) -> "SBMLHandler":
        copy_handler = self.create_copy()
        modified_model: SBMLHandler

        in_rules = False

        # For each Rules in the model, pin the rules to remove
        for rule in self.model.getListOfRules():
            if rule.getVariable() == target_species_id:  # Found the updating rule
                in_rules = True
                # Set the math of the target_species constant to 0
                zero_ast = libsbml.ASTNode(libsbml.AST_INTEGER)
                zero_ast.setValue(0)
                rule.setMath(zero_ast)
                print_log(self.log_file, f"Set rule for {target_species_id} to 0")

        reactions_to_knockout = []

        if not in_rules:
            print_log(
                self.log_file,
                f"Species {target_species_id} not found in rules, looking in reactions...",
            )

            for reaction in copy_handler.model.getListOfReactions():
                reaction_id = reaction.getId()

                should_knockout = False

                for i in range(reaction.getNumReactants()):
                    if reaction.getReactant(i).getSpecies() == target_species_id:
                        should_knockout = True

                        print_log(
                            self.log_file,
                            f"{target_species_id} found as reactant in {reaction_id}",
                        )
                        break

                if should_knockout:
                    reactions_to_knockout.append(reaction_id)

            products_to_remove = []
            for reaction in copy_handler.model.getListOfReactions():
                num_products = reaction.getNumReactants()

                if reaction.getId() not in reactions_to_knockout:
                    for i in range(num_products):
                        if reaction.getProduct(i).getSpecies() == target_species_id:
                            products_to_remove.append((reaction.getId(), i))
                            print_log(
                                self.log_file,
                                f"{target_species_id} found as product in reaction {reaction.getId()}",
                            )

            # Remove products (from highest index to lowest to avoid index shifting)
            products_by_reaction = {}
            for reaction_id, product_idx in products_to_remove:
                if reaction_id not in products_by_reaction:
                    products_by_reaction[reaction_id] = []
                products_by_reaction[reaction_id].append(product_idx)

            for reaction_id, indices in products_by_reaction.items():
                reaction = copy_handler.model.getReaction(reaction_id)

                for idx in sorted(indices, reverse=True):
                    print_log(
                        self.log_file,
                        f"Removing product at index {idx} from {reaction_id}",
                    )
                    reaction.removeProduct(idx)

            # Knockout the reactions that need to be knocked out
            for reaction_id in reactions_to_knockout:
                print_log(self.log_file, f"Calling knockout_reaction on {reaction_id}")
                modified_model = self.knockout_reaction(
                    copy_handler.model, reaction_id, self.log_file
                )

        # Set the initial concentration to 0.0 and make the species constant
        for species in copy_handler.model.getListOfSpecies():
            if species.getId() == target_species_id:
                result = species.setInitialConcentration(0.0)
                result = species.setConstant(True)
                if result == libsbml.LIBSBML_OPERATION_FAILED:
                    print_log(
                        self.log_file,
                        f"Error setting concentration for {species.getId()}",
                    )

        return modified_model

    def generate_species_samples(
        self, target_species: list[str], n_samples: int = 5, variation: float = 20.0
    ) -> List[List[float]]:
        samples = []
        pass

        # TODO: Complete the handler class
