"""Base class for reaction components (reagents and products)."""

class ReactionComponent:
    """
    Base class for reagents and products in a reaction.
    
    Attributes:
        species (Species): The chemical species
        stoichiometry (float): Stoichiometric coefficient
    """
    
    def __init__(self, species, stoichiometry=1.0):
        """
        Initialize a new ReactionComponent.
        
        Args:
            species (Species): The species involved
            stoichiometry (float, optional): Stoichiometric coefficient. Defaults to 1.0.
        """
        self.species = species
        self.stoichiometry = stoichiometry
    
    def __str__(self):
        """Return string representation of this component."""
        coef = "" if self.stoichiometry == 1.0 else f"{self.stoichiometry} "
        return f"{coef}{self.species}"
    
    def to_dict(self):
        """Convert component to dictionary representation."""
        return {
            'species': self.species.to_dict(),
            'stoichiometry': self.stoichiometry
        }