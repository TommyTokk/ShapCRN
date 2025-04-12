"""Reaction class representing a chemical reaction."""

from .reagent import Reagent
from .product import Product

class Reaction:
    """
    A chemical reaction with reagents and products.
    
    Attributes:
        id (str): Unique identifier
        name (str): Human-readable name
        reagents (list): List of Reagent objects
        products (list): List of Product objects
        reversible (bool): Whether the reaction is reversible
        kinetic_law (str): Mathematical expression of the kinetic law
    """
    
    def __init__(self, id, name=None, reagents=None, products=None, 
                 reversible=False, kinetic_law=None):
        """
        Initialize a new Reaction.
        
        Args:
            id (str): Unique identifier
            name (str, optional): Human-readable name
            reagents (list, optional): List of Reagent objects
            products (list, optional): List of Product objects
            reversible (bool, optional): Whether the reaction is reversible
            kinetic_law (str, optional): Mathematical expression of the kinetic law
        """
        self.id = id
        self.name = name or id
        self.reagents = reagents or []
        self.products = products or []
        self.reversible = reversible
        self.kinetic_law = kinetic_law
    
    def __str__(self):
        """Return string representation of this reaction."""
        reagents_str = " + ".join(str(r) for r in self.reagents)
        products_str = " + ".join(str(p) for p in self.products)
        arrow = " <-> " if self.reversible else " -> "
        return f"{reagents_str}{arrow}{products_str}"
    
    def to_dict(self):
        """Convert reaction to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'reagents': [r.to_dict() for r in self.reagents],
            'products': [p.to_dict() for p in self.products],
            'reversible': self.reversible,
            'kinetic_law': self.kinetic_law
        }
    
    @classmethod
    def from_sbml(cls, sbml_reaction, species_dict):
        """
        Create a Reaction from an SBML reaction.
        
        Args:
            sbml_reaction: SBML reaction object
            species_dict: Dictionary of Species objects indexed by ID
            
        Returns:
            Reaction: New Reaction instance
        """
        # Create reagents
        reagents = []
        for reactant in sbml_reaction.getListOfReactants():
            reagents.append(Reagent.from_sbml(reactant, species_dict))
        
        # Create products
        products = []
        for product in sbml_reaction.getListOfProducts():
            products.append(Product.from_sbml(product, species_dict))
        
        # Get kinetic law if present
        kinetic_law = None
        if sbml_reaction.isSetKineticLaw():
            kinetic_law = sbml_reaction.getKineticLaw().getFormula()
            
        return cls(
            id=sbml_reaction.getId(),
            name=sbml_reaction.getName() or sbml_reaction.getId(),
            reagents=reagents,
            products=products,
            reversible=sbml_reaction.getReversible(),
            kinetic_law=kinetic_law
        )
    
    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name
    
    def get_reagents(self):
        return self.reagents
    
    def get_products(self):
        return self.products
    
    def get_reversible(self):
        return self.reversible
    
    def get_kinetic_law(self):
        return self.kinetic_law
