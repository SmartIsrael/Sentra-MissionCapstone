"""
Custom middleware for performance optimization
"""


class CacheControlMiddleware:
    """
    Middleware to add cache control headers to responses
    """

    def __init__(self, get_response):
        self.get_response = get_response
        # Define cache times for different content types (in seconds)
        self.cache_times = {
            "text/html": 3600,  # 1 hour for HTML
            "text/css": 604800,  # 1 week for CSS
            "application/javascript": 604800,  # 1 week for JS
            "image/jpeg": 2592000,  # 30 days for JPEG images
            "image/png": 2592000,  # 30 days for PNG images
            "image/svg+xml": 2592000,  # 30 days for SVG
            "image/webp": 2592000,  # 30 days for WebP
            "font/woff": 2592000,  # 30 days for fonts
            "font/woff2": 2592000,  # 30 days for fonts
        }

    def __call__(self, request):
        # Process the request
        response = self.get_response(request)

        # Skip for admin URLs
        if request.path.startswith("/admin/"):
            return response

        # Add Cache-Control header based on content type
        content_type = response.get("Content-Type", "")
        if content_type:
            content_type = content_type.split(";")[0].strip()

        # Set cache time based on content type
        if content_type in self.cache_times:
            max_age = self.cache_times[content_type]
            response["Cache-Control"] = f"public, max-age={max_age}"

        return response
