from libsbml import formulaToL3String
import libsbml

class Function:
    def __init__(self, id = None, metaId = None, name = None, variables = [], func = None):
        self.setId(id)
        self.setMetaId(metaId)
        self.setName(name)
        self.setVariables(variables)
        self.setFunc(func)

    @classmethod
    def from_sbml(cls, sbml_func):
        """
        Create a Function obj from FunctionDefinition SBML
        
        Args:
            sbml_func: SBML FunctionDefinition object
            
        Returns:
            Function: new Function
        """
        bvar = []
        for i in range(sbml_func.getNumArguments()):
            #print(f"Argument {i}: {formulaToL3String(sbml_func.getArgument(i))}")
            bvar.append(formulaToL3String(sbml_func.getArgument(i)))
        
        
        math_ast = sbml_func.getMath()
        
        if math_ast.getType() == libsbml.AST_LAMBDA:
            nth_children_idx = math_ast.getNumChildren() - 1
            nth_child = math_ast.getChild(nth_children_idx)

            lambda_func = formulaToL3String(nth_child) if nth_child is not None else ""

        else:
            lambda_func = formulaToL3String(math_ast) if math_ast is not None else ""
            print(f"else case lambda_func: {lambda_func}")
        
        return cls(
            id=sbml_func.getId(),
            metaId=sbml_func.getMetaId(),
            name=sbml_func.getName(),
            variables=bvar,
            func=lambda_func
        )
    

    def __str__(self):
        """Return string representation of this function."""
        name = self.getName() if self.getName() is not None else self.getId() if self.getId() is not None else "Unnamed"
        vars_str = ", ".join(v for v in self.getVariables() if v is not None) if self.getVariables() else ""
        func_body = self.getFunc() if self.getFunc() is not None else ""
        
        return f"{name}({vars_str}) = {func_body}"
    

    def to_dict(self):
        return {
            'id': self.getId(),
            'metaId': self.getMetaId(),
            'name': self.getName(),
            'variables': self.getVariables(),
            'lambda_func': self.getFunc()
        }
    
    #GETTERS
    def getId(self):
        return self.id
    
    def getMetaId(self):
        return self.metaId 
    
    def getName(self):
        return self.name 
    
    def getVariables(self):
        return self.variables 
    
    def getFunc(self):
        return self.func
    
    #GETTERS


    #SETTERS
    def setId(self, id):
        self.id = id if id is not None else ""

    def setMetaId(self, metaId):
        self.metaId = metaId if metaId is not None else ""
    
    def setName(self, name):
        self.name = name if name is not None else ""

    def setVariables(self, variables):
        if variables is None:
            self.variables = []
        else:
            # Assicurati che non ci siano valori None nella lista
            self.variables = [v for v in variables if v is not None]

    def setFunc(self, func):
        self.func = func if func is not None else ""
    #SETTERS

