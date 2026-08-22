from urllib.parse import quote_plus


def get_ecommerce_links(cloth_type, cloth_color):
    # Strip "closest match: " prefix if present
    color = cloth_color.replace('closest match: ', '').strip()

    # Search query uses only the clothing type
    query = quote_plus(cloth_type)

    links = {
        "type" : cloth_type,
        "color": color,
        "stores": {
            "daraz"   : f"https://www.daraz.pk/catalog/?q={query}",
            "goto"    : f"https://www.goto.com.pk/search?q={query}",
            "laam"    : f"https://laam.pk/search?q={query}",
            "sapphire": f"https://pk.sapphireonline.pk/search?q={query}",
            "khaadi"  : f"https://www.khaadi.com/search?q={query}",
            "limelight": f"https://www.limelight.pk/search?q={query}"
        }
    }

    return links
