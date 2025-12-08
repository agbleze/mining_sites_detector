


def create_alteration_index(band1: str, band2: str) -> str:
    """Create a string representation of an alteration index formula.

    Args:
        band1: The name of the first spectral band.
        band2: The name of the second spectral band.

    Returns:
        A string representing the alteration index formula.
    """
    return f"({band1} - {band2}) / ({band1} + {band2})"


"""
	Alteration		
1600:1700
2145:2185
 
11
12
 
Automatic
"""