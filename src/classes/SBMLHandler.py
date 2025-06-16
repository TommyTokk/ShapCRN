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

        self._log(f"{model_path}")

        if model_path:
            self.load_model(model_path)

    # === MODEL LOADING ===
    def load_model(self, model_path: str) -> bool:
        try:
            reader = libsbml.SBMLReader()
            self.document = reader.readSBML(model_path)

            if self.document.getNumErrors() > 0:
                self._log("Error during model loading")
                return False

            self.model = self.document.getModel()
            self._model_path = model_path

            self._log(f"Successfully loaded SBML model: {model_path}")
            return True

        except Exception as e:
            self._log(f"Error loading SBML model: {e}")
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
            self._log(f"Error loading from string: {e}")
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

    def generate_species_samples(
        self, target_species: list[str], n_samples: int = 5, variation: float = 20.0
    ) -> List[List[float]]:
        samples = []

        for ts in target_species:
            t0_conc = (
                self.model.getListOfSpecies()
                .getElementById(ts)
                .getInitialConcentration()
            )

            ts_samples = []

            for i in range(n_samples):
                factor = np.random.uniform(1 - variation / 100, 1 + variation / 100)
                sample = t0_conc + factor
                ts_samples.append(sample)

            samples.append(ts_samples)

        return samples

    def knockout_species(self, target_species_id: str) -> "SBMLHandler":
        copy_handler = self.create_copy()

        in_rules = False

        # For each Rules in the model, pin the rules to remove
        for rule in self.model.getListOfRules():
            if rule.getVariable() == target_species_id:  # Found the updating rule
                in_rules = True
                # Set the math of the target_species constant to 0
                zero_ast = libsbml.ASTNode(libsbml.AST_INTEGER)
                zero_ast.setValue(0)
                rule.setMath(zero_ast)
                self._log(f"Set rule for {target_species_id} to 0")

        reactions_to_knockout = []

        if not in_rules:
            self._log(
                f"Species {target_species_id} not found in rules, looking in reactions...",
            )

            for reaction in copy_handler.model.getListOfReactions():
                reaction_id = reaction.getId()

                should_knockout = False

                for i in range(reaction.getNumReactants()):
                    if reaction.getReactant(i).getSpecies() == target_species_id:
                        should_knockout = True

                        self._log(
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
                            self._log(
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
                    self._log(
                        f"Removing product at index {idx} from {reaction_id}",
                    )
                    reaction.removeProduct(idx)

            # Knockout the reactions that need to be knocked out
            for reaction_id in reactions_to_knockout:
                reaction = copy_handler.model.getReaction(reaction_id)
                self.set_reaction_kinetic_zero(reaction)

        # Set the initial concentration to 0.0 and make the species constant
        for species in copy_handler.model.getListOfSpecies():
            if species.getId() == target_species_id:
                result = species.setInitialConcentration(0.0)
                result = species.setConstant(True)
                if result == libsbml.LIBSBML_OPERATION_FAILED:
                    self._log(
                        f"Error setting concentration for {species.getId()}",
                    )

        return copy_handler

    # === REACTION MANAGEMENT ===

    def split_reversible_reaction(self, reaction_id: str):

        # TODO: Implementare lo split in modo corretto

        self._log(f"{reaction_id}")
        reaction = self.model.getReaction(reaction_id)

        # Get the KL of the reaction
        kinetic_law = reaction.getKineticLaw()

        if kinetic_law is None:
            raise ValueError(
                f"Reaction '{reaction_id}' does not have a kinetic law defined."
            )

        # Get the parameters of the reaction
        parameters = {}

        # First search in the reaction's parameters
        n_params = kinetic_law.getNumParameters()

        if n_params != 0:
            for i in range(n_params):
                param = kinetic_law.getParameter(i)
                parameters[param.getId()] = param.getValue()
        else:
            # Search the parameters in the model
            n_model_params = self.model.getNumParameters()

            if n_model_params != 0:
                for i in range(n_model_params):
                    model_param = self.model.getParameter(i)
                    parameters[model_param.getId()] = model_param.getValue()

        self._log(f"{parameters}")

        kl_math = kinetic_law.getMath()

        nodes = self.parse_ast_tree(kl_math)

        self._log(f"{nodes}")

    def create_reaction(
        self,
        reaction_name,
        reaction_id,
        reaction_tags,
        reaction_prods,
        reaction_react,
        reaction_parameters,
    ):

        reaction = self.model.createReaction()
        pass

    def knockout_reaction(self, target_reaction_id: str) -> "SBMLHandler":

        copy_handler = self.create_copy()
        reaction = self.model.getReaction(target_reaction_id)

        if reaction:
            success = self.set_reaction_kinetic_zero(reaction)
            if success:
                self._log(f"Knocked out reaction {target_reaction_id}")
            else:
                self._log(f"Failed to knockout reaction {target_reaction_id}")
        return copy_handler

    # === UTILS ===
    def set_species_concentration_zero(self, model, species_id: str) -> bool:
        species = model.getSpecies(species_id)
        if species:
            species.setInitialConcentration(0.0)
            species.setConstant(True)
            self._log(f"Set {species_id} concentration to 0 and constant=True")
            return True
        return False

    def set_reaction_kinetic_zero(self, reaction) -> bool:
        if reaction is not None:
            zero_ast = libsbml.ASTNode(libsbml.AST_INTEGER)
            zero_ast.setValue(0)
            kinetic_law = reaction.getKineticLaw()
            if kinetic_law:
                result = kinetic_law.setMath(zero_ast)
                return result == libsbml.LIBSBML_OPERATION_SUCCESS
        return False

    def parse_ast_tree(self, ast_node):
        """
        Recursively parses an ASTNode from libSBML and returns a dictionary representation.
        """

        if ast_node is None:
            return None

        # Base case: if node is a leaf (a number or a variable)
        if ast_node.getNumChildren() == 0:
            if ast_node.isName():
                return {"type": "name", "value": ast_node.getName()}
            elif ast_node.isNumber():
                return {"type": "number", "value": ast_node.getValue()}
            else:
                return {"type": "unknown", "value": ast_node.toFormula()}

        operator = ast_node.getName()
        children = [
            self.parse_ast_tree(ast_node.getChild(i))
            for i in range(ast_node.getNumChildren())
        ]

        return {
            "type": "operator",
            "op": operator,
            "op_type": ast_node.getType(),
            "args": children,
        }

    # === PRIVATE ===

    def _log(self, msg_str: str):
        current_date = datetime.datetime.now()
        if self.log_file:
            with open(self.log_file, "a") as out:
                out.write(f"[{current_date}]: {msg_str}\n")
        else:
            print(f"[{current_date}]: {msg_str}")
