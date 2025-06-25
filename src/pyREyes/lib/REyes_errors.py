# Custom Exceptions
class GridSquareError(Exception):
    """Base exception for grid square processing errors."""
    pass

class CentroidDetectionError(GridSquareError):
    """Exception raised when centroid detection fails."""
    pass

class FileNotFoundError(GridSquareError):
    """Exception raised when required files are not found."""
    pass



class ProcessingError(Exception):
    """Custom exception for processing-related errors."""
    pass