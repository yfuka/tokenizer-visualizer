import hashlib

def string_to_color(text: str, saturation: int = 70, lightness: int = 80) -> str:
    """
    Generates a consistent HSL color string for a given text.
    
    Args:
        text: The input text (e.g., token string or ID).
        saturation: Saturation percentage (0-100).
        lightness: Lightness percentage (0-100).
        
    Returns:
        A CSS HSL color string (e.g., "hsl(120, 70%, 80%)").
    """
    # Use MD5 to get a consistent hash
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    
    # Take the first few bytes to determine hue (0-360)
    hue = int(hash_hex[:4], 16) % 360
    
    return f"hsl({hue}, {saturation}%, {lightness}%)"

def get_token_type_color(token_type: str) -> str:
    """
    Returns a specific color for special token types if needed.
    """
    # Placeholder for future special token coloring
    return "#e0e0e0"
