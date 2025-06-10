/**
 * Image Optimizer Script
 * Optimizes image loading by implementing:
 * 1. Native lazy loading for below-the-fold images
 * 2. Progressive image loading
 * 3. Image preloading when idle
 */

(function() {
    // Use requestIdleCallback or setTimeout for non-critical image loading
    const optimizeImages = function() {
        // Add lazy loading to all images except the critical ones
        const allImages = document.querySelectorAll('img:not([fetchpriority="high"])');
        let imagesProcessed = 0;
        
        // Process images in batches to avoid blocking the main thread
        function processNextBatchOfImages(deadline) {
            while (imagesProcessed < allImages.length && 
                   (deadline.timeRemaining() > 0 || deadline.didTimeout)) {
                const img = allImages[imagesProcessed];
                
                // Skip already processed images
                if (img.hasAttribute('data-processed')) {
                    imagesProcessed++;
                    continue;
                }
                
                // Check if image is below the fold (not in viewport)
                const rect = img.getBoundingClientRect();
                const isBelowFold = rect.top > window.innerHeight;
                
                // Apply lazy loading for non-critical images
                if (isBelowFold && !img.hasAttribute('loading')) {
                    img.setAttribute('loading', 'lazy');
                }
                
                // Mark as processed to avoid reprocessing
                img.setAttribute('data-processed', 'true');
                imagesProcessed++;
            }
            
            // If there are more images to process, continue in the next idle period
            if (imagesProcessed < allImages.length) {
                requestIdleCallback(processNextBatchOfImages);
            }
        }
        
        // Start processing images
        if ('requestIdleCallback' in window) {
            requestIdleCallback(processNextBatchOfImages);
        } else {
            // Fallback for browsers without requestIdleCallback
            setTimeout(function() {
                allImages.forEach(img => {
                    if (!img.hasAttribute('loading')) {
                        img.setAttribute('loading', 'lazy');
                    }
                });
            }, 200);
        }
        
        // Preload critical images for the next section when user scrolls near them
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                // When a section comes close to viewport, preload its images
                if (entry.isIntersecting) {
                    const nextSection = entry.target.nextElementSibling;
                    if (nextSection) {
                        const nextImages = nextSection.querySelectorAll('img[loading="lazy"]');
                        if (nextImages.length) {
                            // Use requestIdleCallback to preload without blocking
                            if ('requestIdleCallback' in window) {
                                requestIdleCallback(() => {
                                    nextImages.forEach(img => {
                                        const src = img.getAttribute('src');
                                        if (src) {
                                            const preloadImg = new Image();
                                            preloadImg.src = src;
                                        }
                                    });
                                });
                            }
                        }
                    }
                    observer.unobserve(entry.target);
                }
            });
        }, { rootMargin: '200px' }); // Start preloading when within 200px
        
        // Observe sections for preloading next section's images
        document.querySelectorAll('section').forEach(section => {
            observer.observe(section);
        });
    };
    
    // Run on DOMContentLoaded to ensure the DOM is available
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            // Let critical content load first, then optimize images
            setTimeout(optimizeImages, 100);
        });
    } else {
        // DOM already loaded, run with small delay
        setTimeout(optimizeImages, 100);
    }
})();