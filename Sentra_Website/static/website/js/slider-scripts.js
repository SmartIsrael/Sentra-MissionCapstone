/**
 * Optimized JavaScript for Smartel's HTML Slider
 * Performance-enhanced version that reduces runtime overhead
 */

// Use requestIdleCallback or setTimeout for non-critical initialization
(function() {
    // Wait for DOM to be interactive before setting up slider
    const setupSlider = function() {
        // Cache DOM elements to avoid repeated queries
        const slides = document.querySelectorAll('.smartel-slide');
        const sliderContainer = document.querySelector('.smartel-slider');
        const prevBtn = document.getElementById('slider-prev');
        const nextBtn = document.getElementById('slider-next');
        
        // Don't initialize if elements aren't found
        if (!slides.length || !sliderContainer || !prevBtn || !nextBtn) return;
        
        let currentSlide = 0;
        const slideCount = slides.length;
        let autoSlideInterval = null;
        let isVisible = true;
        
        // Use Page Visibility API to pause when tab not active
        document.addEventListener('visibilitychange', function() {
            isVisible = document.visibilityState === 'visible';
            if (isVisible) {
                startAutoSlide();
            } else {
                stopAutoSlide();
            }
        });
        
        // Use more efficient event binding
        const nextSlide = function() {
            if (slideCount <= 1) return; // Don't animate if only one slide
            slides[currentSlide].classList.remove('active');
            currentSlide = (currentSlide + 1) % slideCount;
            slides[currentSlide].classList.add('active');
        };
        
        const prevSlide = function() {
            if (slideCount <= 1) return; // Don't animate if only one slide
            slides[currentSlide].classList.remove('active');
            currentSlide = (currentSlide - 1 + slideCount) % slideCount;
            slides[currentSlide].classList.add('active');
        };
        
        // Auto-slide with performance optimizations
        const startAutoSlide = function() {
            stopAutoSlide();
            if (slideCount > 1 && isVisible) {
                autoSlideInterval = setTimeout(function autoSlide() {
                    nextSlide();
                    autoSlideInterval = setTimeout(autoSlide, 7000);
                }, 7000);
            }
        };
        
        const stopAutoSlide = function() {
            if (autoSlideInterval) {
                clearTimeout(autoSlideInterval);
                autoSlideInterval = null;
            }
        };
        
        // Add event listeners with passive option for touch events
        prevBtn.addEventListener('click', prevSlide);
        nextBtn.addEventListener('click', nextSlide);
        sliderContainer.addEventListener('mouseenter', stopAutoSlide, {passive: true});
        sliderContainer.addEventListener('mouseleave', startAutoSlide, {passive: true});
        
        // Ensure first slide is active
        if (slides[0] && !slides[0].classList.contains('active')) {
            slides[0].classList.add('active');
        }
        
        // Start auto-slide
        startAutoSlide();
        
        // Preload next slide image when idle
        if ('requestIdleCallback' in window) {
            requestIdleCallback(function() {
                const images = document.querySelectorAll('.smartel-slide:not(.active) img');
                if (images.length) {
                    images.forEach(img => {
                        if (img.getAttribute('loading') === 'lazy') {
                            // Force load the image
                            const src = img.getAttribute('src');
                            if (src) {
                                const preloadImg = new Image();
                                preloadImg.src = src;
                            }
                        }
                    });
                }
            });
        }
    };
    
    // Use requestIdleCallback if available, or fallback to setTimeout
    if ('requestIdleCallback' in window) {
        requestIdleCallback(function() {
            setupSlider();
        });
    } else {
        // Use a small delay to allow critical content to load first
        setTimeout(setupSlider, 100);
    }
})();
