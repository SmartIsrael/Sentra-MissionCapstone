// Enhanced image sections JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Add intersection observer for fade-in effects
    const imageElements = document.querySelectorAll('.pq-image-1, .pq-image-2');
    
    // Add enhanced classes to image container elements
    const aboutMediaSections = document.querySelectorAll('.pq-about-media');
    aboutMediaSections.forEach(section => {
        section.classList.add('enhanced-image-layout');
        
        // Add accent elements
        const accent1 = document.createElement('div');
        accent1.classList.add('image-accent', 'accent-1');
        section.appendChild(accent1);
        
        const accent2 = document.createElement('div');
        accent2.classList.add('image-accent', 'accent-2');
        section.appendChild(accent2);
    });
    
    // Add parallax scroll effect to images
    window.addEventListener('scroll', function() {
        const scrollPosition = window.scrollY;
        
        imageElements.forEach(img => {
            const parent = img.closest('section');
            const parentTop = parent.offsetTop;
            const parentHeight = parent.offsetHeight;
            
            // Check if image is in viewport
            if (scrollPosition > parentTop - window.innerHeight && 
                scrollPosition < parentTop + parentHeight) {
                
                // Calculate parallax offset (subtle effect)
                const scrollPercentage = (scrollPosition - (parentTop - window.innerHeight)) / 
                                        (parentHeight + window.innerHeight);
                const moveY = scrollPercentage * 30; // 30px max movement
                
                // Apply parallax effect
                if (img.classList.contains('pq-image-1')) {
                    img.style.transform = `translateY(${-moveY}px)`;
                } else {
                    img.style.transform = `translateY(${moveY}px)`;
                }
            }
        });
    });
    
    // Add conditional load animation
    function animateOnScroll() {
        aboutMediaSections.forEach(section => {
            const rect = section.getBoundingClientRect();
            const isInViewport = (
                rect.top <= (window.innerHeight || document.documentElement.clientHeight) * 0.8 &&
                rect.bottom >= 0
            );
            
            if (isInViewport) {
                section.classList.add('animate-in');
            }
        });
    }
    
    // Run animation check on scroll
    window.addEventListener('scroll', animateOnScroll);
    // Initial check
    animateOnScroll();
    
    // Optional: Add image loading optimization
    if ('loading' in HTMLImageElement.prototype) {
        imageElements.forEach(img => {
            img.loading = 'lazy';
        });
    }
});
