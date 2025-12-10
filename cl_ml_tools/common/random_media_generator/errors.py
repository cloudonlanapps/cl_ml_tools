class JSONValidationError(Exception):
    """
    A custom exception class for demonstrating custom error handling.
    This exception can be raised when a specific condition is not met.
    """

    def __init__(self, message="An unknown custom error occurred."):
        # Call the base class constructor with the custom message
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        # Define how the exception object should be represented as a string
        return f"CustomError: {self.message}"