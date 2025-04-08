"""Reagent class representing a reactant in a chemical reaction."""

from .component import ReactionComponent

class Reagent(ReactionComponent):
    """
    A reagent (reactant) in a chemical reaction.
    
    Inherits from:
        ReactionComponent: Base class for reaction components
    """
    
    def __init__(self, species, stoichiometry=1.0):
        """
        Initialize a new Reagent.
        
        Args:
            species (Species): The species that acts as reactant
            stoichiometry (float, optional): Stoichiometric coefficient. Defaults to 1.0.
        """
        super().__init__(species, stoichiometry)
    
    @classmethod
    def from_sbml(cls, sbml_reactant, species_dict):
        """
        Create a Reagent from an SBML reactant reference.
        
        Args:
            sbml_reactant: SBML reactant reference
            species_dict: Dictionary of Species objects indexed by ID
            
        Returns:
            Reagent: New Reagent instance
        """
        species_id = sbml_reactant.getSpecies()
        if species_id not in species_dict:
            raise ValueError(f"Species {species_id} not found in species dictionary")
            
        return cls(
            species=species_dict[species_id],
            stoichiometry=sbml_reactant.getStoichiometry()
        )