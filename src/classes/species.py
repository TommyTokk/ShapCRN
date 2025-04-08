"""Species class representing a chemical species in a reaction network."""

class Species:
    """
    A chemical species in a reaction network.
    
    Attributes:
        id (str): Unique identifier for this species
        name (str): Human-readable name
        compartment (str): Compartment where this species is located
        initial_amount (float): Initial amount of this species
    """
    
    def __init__(self, id, name=None, compartment=None, initial_concentration=0.0):
        """
        Initialize a new Species.
        
        Args:
            id (str): Unique identifier
            name (str, optional): Human-readable name
            compartment (str, optional): Compartment where this species is located
            initial_amount (float, optional): Initial amount. Defaults to 0.0.
        """
        self.set_id(id)
        self.set_name(name)
        self.set_compartment(compartment)
        self.set_initial_concentration(initial_concentration)
    
    def __str__(self):
        """Return string representation of this species."""
        return f"{self.name} ({self.id})"
    
    def to_dict(self):
        """Convert species to dictionary representation."""
        return {
            'id': self.get_id(),
            'name': self.get_name(),
            'compartment': self.get_compartment(),
            'initial_concentration': self.get_initial_concentration()
        }
    
    @classmethod
    def from_sbml(cls, sbml_species):
        """
        Create a Species object from an SBML species element.
        
        Args:
            sbml_species: SBML species object
            
        Returns:
            Species: New Species instance
        """
        return cls(
            id=sbml_species.getId(),
            name=sbml_species.getName() or sbml_species.getId(),
            compartment=sbml_species.getCompartment(),
            initial_concentration=sbml_species.getInitialConcentration()
        )
    
    """
    GETTERS
    """
    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name
    
    def get_compartment(self):
        return self.compartment
    
    def get_initial_concentration(self):
        return self.initial_concentration
    

    """
    SETTERS
    """
    
    def set_id(self, id):
        self.id = id
    
    def set_name(self, name):
        self.name = name
    
    def set_compartment(self, comp):
        self.compartment = comp 

    def set_initial_concentration(self, initial_concentration):
        self.initial_concentration = initial_concentration

