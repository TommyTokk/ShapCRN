"""Custom exceptions for SBML model analysis and Shapley value computation."""


# ============================================================================
# BASE EXCEPTION
# ============================================================================


class KOShapleyError(Exception):
    """Base exception for KO Shapley Value analysis."""

    pass


# ============================================================================
# MODEL-RELATED EXCEPTIONS
# ============================================================================


class ModelError(KOShapleyError):
    """Base exception for SBML model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when an SBML model file cannot be found."""

    def __init__(self, model_path, message=None):
        self.model_path = model_path
        if message is None:
            message = f"SBML model file not found: {model_path}"
        super().__init__(message)


class InvalidModelFormatError(ModelError):
    """Raised when SBML model format is invalid or cannot be parsed."""

    def __init__(self, model_path, details=None):
        self.model_path = model_path
        self.details = details
        message = f"Invalid SBML model format: {model_path}"
        if details:
            message += f"\nDetails: {details}"
        super().__init__(message)


class InvalidSpeciesError(ModelError):
    """Raised when a species ID is invalid or not found in the model."""

    def __init__(self, species_id, model_id=None, message=None):
        self.species_id = species_id
        self.model_id = model_id
        if message is None:
            message = f"Species '{species_id}' not found in model"
            if model_id:
                message += f" '{model_id}'"
        super().__init__(message)


class InvalidReactionError(ModelError):
    """Raised when a reaction ID is invalid or not found in the model."""

    def __init__(self, reaction_id, model_id=None, message=None):
        self.reaction_id = reaction_id
        self.model_id = model_id
        if message is None:
            message = f"Reaction '{reaction_id}' not found in model"
            if model_id:
                message += f" '{model_id}'"
        super().__init__(message)


class InvalidKineticLawError(ModelError):
    """Raised when the reaction has not a defined KineticLaw node"""

    def __init__(self, reaction_id: str, message=None) -> None:
        self.reaction_id = reaction_id
        if message is None:
            message = f"Reaction {reaction_id} has not a defined KineticLaw node"

        super().__init__(message)


class InvalidFunctionDefinitionError(ModelError):
    """Raised when the function has not been correctly defined"""

    def __init__(self, function_id: str, message=None) -> None:
        self.function_id = function_id
        if message is None:
            message = f"Function {function_id} has not been defined correctly."
        super().__init__(message)


class ModelModificationError(ModelError):
    """Raised when model modification (knockout/knockin) fails."""

    def __init__(self, operation, target, reason=None):
        self.operation = operation
        self.target = target
        self.reason = reason
        message = f"Failed to {operation} target '{target}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


# ============================================================================
# SIMULATION-RELATED EXCEPTIONS
# ============================================================================


class SimulationError(KOShapleyError):
    """Base exception for simulation-related errors."""

    pass


class SimulationFailedError(SimulationError):
    """Raised when RoadRunner simulation fails."""

    def __init__(self, model_id=None, details=None):
        self.model_id = model_id
        self.details = details
        message = "Simulation failed"
        if model_id:
            message += f" for model '{model_id}'"
        if details:
            message += f": {details}"
        super().__init__(message)


class SteadyStateNotReachedError(SimulationError):
    """Raised when steady state is not reached within specified time."""

    def __init__(self, max_time, threshold, message=None):
        self.max_time = max_time
        self.threshold = threshold
        if message is None:
            message = (
                f"Steady state not reached within {max_time} time units "
                f"(threshold: {threshold})"
            )
        super().__init__(message)


class InvalidIntegratorError(SimulationError):
    """Raised when an invalid integrator is specified."""

    def __init__(self, integrator, valid_integrators=None):
        self.integrator = integrator
        self.valid_integrators = valid_integrators
        message = f"Invalid integrator: '{integrator}'"
        if valid_integrators:
            message += f". Valid options: {', '.join(valid_integrators)}"
        super().__init__(message)


# ============================================================================
# ANALYSIS-RELATED EXCEPTIONS
# ============================================================================


class AnalysisError(KOShapleyError):
    """Base exception for analysis-related errors."""

    pass


class ConvergenceError(AnalysisError):
    """Raised when sensitivity analysis fails to converge."""

    def __init__(self, node, max_samples, message=None):
        self.node = node
        self.max_samples = max_samples
        if message is None:
            message = f"Node '{node}' failed to converge within {max_samples} samples"
        super().__init__(message)


class InsufficientSamplesError(AnalysisError):
    """Raised when not enough samples are provided for analysis."""

    def __init__(self, provided, required, message=None):
        self.provided = provided
        self.required = required
        if message is None:
            message = (
                f"Insufficient samples for analysis: "
                f"{provided} provided, {required} required"
            )
        super().__init__(message)


class InvalidSobolParametersError(AnalysisError):
    """Raised when Sobol analysis parameters are invalid."""

    def __init__(self, parameter, value, reason=None):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        message = f"Invalid Sobol parameter '{parameter}': {value}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class ShapleyValueComputationError(AnalysisError):
    """Raised when Shapley value computation fails."""

    def __init__(self, target_node=None, details=None):
        self.target_node = target_node
        self.details = details
        message = "Shapley value computation failed"
        if target_node:
            message += f" for target node '{target_node}'"
        if details:
            message += f": {details}"
        super().__init__(message)


# ============================================================================
# NETWORK-RELATED EXCEPTIONS
# ============================================================================


class NetworkError(KOShapleyError):
    """Base exception for network-related errors."""

    pass


class NetworkConstructionError(NetworkError):
    """Raised when network construction from SBML model fails."""

    def __init__(self, model_id=None, reason=None):
        self.model_id = model_id
        self.reason = reason
        message = "Failed to construct network from SBML model"
        if model_id:
            message += f" '{model_id}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class InvalidNodeError(NetworkError):
    """Raised when a node is not found in the network."""

    def __init__(self, node_id, node_type=None):
        self.node_id = node_id
        self.node_type = node_type
        message = f"Node '{node_id}' not found in network"
        if node_type:
            message += f" (expected type: {node_type})"
        super().__init__(message)


class NetworkVisualizationError(NetworkError):
    """Raised when network visualization fails."""

    def __init__(self, reason=None):
        self.reason = reason
        message = "Network visualization failed"
        if reason:
            message += f": {reason}"
        super().__init__(message)


# ============================================================================
# VALIDATION EXCEPTION
# ============================================================================


class ValidationError(KOShapleyError):
    """Raised when validation fails with detailed context."""

    def __init__(self, errors, context=None, message="Validation failed"):
        """
        Initialize validation error.

        Parameters
        ----------
        errors : list of str or list of dict
            List of validation errors. Can be simple strings or dictionaries
            with keys like 'field', 'error', 'value'.
        context : str, optional
            Additional context about what was being validated.
        message : str, optional
            Main error message, by default "Validation failed"
        """
        self.errors = errors if isinstance(errors, list) else [errors]
        self.context = context
        self.message = message

        # Build full message
        full_message = f"{message}: {len(self.errors)} error(s) found"
        if context:
            full_message += f" in {context}"

        super().__init__(full_message)

    def __str__(self):
        """Format validation errors as a readable string."""
        lines = [f"{self.message}"]

        if self.context:
            lines.append(f"Context: {self.context}")

        lines.append(f"\nErrors ({len(self.errors)}):")

        for i, error in enumerate(self.errors, 1):
            if isinstance(error, dict):
                # Structured error
                field = error.get("field", "unknown")
                err_msg = error.get("error", "validation failed")
                value = error.get("value")

                error_str = f"  {i}. [{field}] {err_msg}"
                if value is not None:
                    error_str += f" (value: {value})"
                lines.append(error_str)
            else:
                # Simple string error
                lines.append(f"  {i}. {error}")

        return "\n".join(lines)

    def add_error(self, error):
        """Add an additional error to the list."""
        self.errors.append(error)

    def has_field_error(self, field):
        """Check if a specific field has an error."""
        for error in self.errors:
            if isinstance(error, dict) and error.get("field") == field:
                return True
        return False

    def get_field_errors(self, field):
        """Get all errors for a specific field."""
        return [
            error
            for error in self.errors
            if isinstance(error, dict) and error.get("field") == field
        ]


# ============================================================================
# FILE I/O EXCEPTIONS
# ============================================================================


class FileIOError(KOShapleyError):
    """Base exception for file input/output errors."""

    pass


class ReportGenerationError(FileIOError):
    """Raised when report generation fails."""

    def __init__(self, report_type, output_path=None, reason=None):
        self.report_type = report_type
        self.output_path = output_path
        self.reason = reason
        message = f"Failed to generate {report_type} report"
        if output_path:
            message += f" at '{output_path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class InvalidOutputPathError(FileIOError):
    """Raised when output path is invalid or not writable."""

    def __init__(self, path, reason=None):
        self.path = path
        self.reason = reason
        message = f"Invalid output path: {path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)
