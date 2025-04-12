"""Product class representing a product in a chemical reaction."""

from .component import ReactionComponent

class Product(ReactionComponent):
    """
    A product in a chemical reaction.
    
    Inherits from:
        ReactionComponent: Base class for reaction components
    """
    
    def __init__(self, species, stoichiometry=1.0):
        """
        Initialize a new Product.
        
        Args:
            species (Species): The species that is produced
            stoichiometry (float, optional): Stoichiometric coefficient. Defaults to 1.0.
        """
        super().__init__(species, stoichiometry)
    
    @classmethod
    def from_sbml(cls, sbml_product, species_dict):
        """
        Create a Product from an SBML product reference.
        
        Args:
            sbml_product: SBML product reference
            species_dict: Dictionary of Species objects indexed by ID
            
        Returns:
            Product: New Product instance
        """
        species_id = sbml_product.getSpecies()
        if species_id not in species_dict:
            raise ValueError(f"Species {species_id} not found in species dictionary")
            
        return cls(
            species=species_dict[species_id],
            stoichiometry=sbml_product.getStoichiometry()
        )
    
    def get_species(self):
        return self.species
    
    def get_stoichiometry(self):
        return self.stoichiometry