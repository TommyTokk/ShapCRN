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
    
    def get_reagents_ids(self):
        res = []

        for reagents in self.get_reagents():
            res.append(reagents.get_species()['id'])

        return res
    
    def get_products(self):
        return self.products
    
    def get_products_ids(self):
        res = []

        for product in self.get_products():
            res.append(product.get_species()['id'])

        return res
    
    def get_reversible(self):
        return self.reversible
    
    def get_kinetic_law(self):
        return self.kinetic_law
    
    def set_id(self, id):
        """
        Imposta l'ID della reazione.
        
        Args:
            id (str): Il nuovo ID della reazione
        """
        self.id = id
    
    def set_name(self, name):
        """
        Imposta il nome della reazione.
        
        Args:
            name (str): Il nuovo nome della reazione
        """
        self.name = name
    
    def set_reagents(self, reagents):
        """
        Imposta la lista dei reagenti.
        
        Args:
            reagents (list): Lista di oggetti Reagent
        """
        if not isinstance(reagents, list):
            raise TypeError("Il parametro reagents deve essere una lista")
        self.reagents = reagents
    
    def set_products(self, products):
        """
        Imposta la lista dei prodotti.
        
        Args:
            products (list): Lista di oggetti Product
        """
        if not isinstance(products, list):
            raise TypeError("Il parametro products deve essere una lista")
        self.products = products
    
    def add_reagent(self, reagent):
        """
        Aggiunge un reagente alla reazione.
        
        Args:
            reagent (Reagent): Reagente da aggiungere
        """
        from .reagent import Reagent
        if not isinstance(reagent, Reagent):
            raise TypeError("Il parametro deve essere un oggetto di tipo Reagent")
        self.reagents.append(reagent)
    
    def add_product(self, product):
        """
        Aggiunge un prodotto alla reazione.
        
        Args:
            product (Product): Prodotto da aggiungere
        """
        from .product import Product
        if not isinstance(product, Product):
            raise TypeError("Il parametro deve essere un oggetto di tipo Product")
        self.products.append(product)
    
    def remove_reagent(self, reagent_id):
        """
        Rimuove un reagente dalla reazione in base all'ID.
        
        Args:
            reagent_id (str): ID del reagente da rimuovere
        
        Returns:
            bool: True se il reagente è stato rimosso, False altrimenti
        """
        for i, r in enumerate(self.reagents):
            if r.get_species()['id'] == reagent_id:
                del self.reagents[i]
                return True
        return False
    
    def remove_product(self, product_id):
        """
        Rimuove un prodotto dalla reazione in base all'ID.
        
        Args:
            product_id (str): ID del prodotto da rimuovere
        
        Returns:
            bool: True se il prodotto è stato rimosso, False altrimenti
        """
        for i, p in enumerate(self.products):
            if p.get_species()['id'] == product_id:
                del self.products[i]
                return True
        return False
    
    def set_reversible(self, reversible):
        """
        Imposta se la reazione è reversibile.
        
        Args:
            reversible (bool): True se la reazione è reversibile, False altrimenti
        """
        self.reversible = bool(reversible)
    
    def set_kinetic_law(self, kinetic_law):
        """
        Imposta la legge cinetica della reazione.
        
        Args:
            kinetic_law (str): Espressione matematica della legge cinetica
        """
        self.kinetic_law = kinetic_law


