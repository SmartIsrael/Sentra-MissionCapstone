/**
 * Priority Loader
 * Prioritizes the loading of DOM elements based on their importance to user experience
 * Implements a phased loading approach aligned with Core Web Vitals
 */

(function() {
    'use strict';
    
    // Set rendering priorities as early as possible
    document.addEventListener('DOMContentLoaded', function() {
        // Define priority levels
        const priorities = {
            CRITICAL: 1,  // Header, Navigation, Hero section, LCP candidate
            HIGH: 2,      // Above the fold content after LCP
            MEDIUM: 3,    // Just below the fold
            LOW: 4,       // Well below the fold (footer, etc)
            DEFER: 5      // Non-essential (analytics, etc)
        };
        
        // DOM sections with their priorities
        const prioritySections = [
            { selector: 'header, .pq-header-style-1', priority: priorities.CRITICAL },
            { selector: '.pq-bottom-header', priority: priorities.CRITICAL },
            { selector: '.banner, .hero-section, .smartel-slider', priority: priorities.CRITICAL },
            { selector: 'main > section:first-child, #about', priority: priorities.HIGH },
            { selector: 'main > section:nth-child(2), main > section:nth-child(3)', priority: priorities.MEDIUM },
            { selector: 'footer, #pq-footer', priority: priorities.LOW },
            { selector: '.social-widgets, .cookie-notice', priority: priorities.DEFER }
        ];
        
        // Apply resource hints based on priority
        prioritySections.forEach(({ selector, priority }) => {
            const elements = document.querySelectorAll(selector);
            
            elements.forEach(element => {
                // Set data attribute for debugging and styling
                element.dataset.renderPriority = priority;
                
                // Apply optimizations based on priority
                if (priority === priorities.CRITICAL) {
                    // Force images to load early
                    element.querySelectorAll('img').forEach(img => {
                        img.loading = 'eager';
                        img.fetchpriority = 'high';
                    });
                    
                    // Force important background images to load early
                    applyBackgroundImagePriority(element, 'high');
                    
                } else if (priority === priorities.HIGH) {
                    // Use standard priority but ensure loading
                    element.querySelectorAll('img').forEach(img => {
                        if (!img.loading) img.loading = 'eager';
                    });
                    
                    applyBackgroundImagePriority(element, 'auto');
                    
                } else {
                    // Use lazy loading for lower priority elements
                    element.querySelectorAll('img').forEach(img => {
                        if (!img.loading) img.loading = 'lazy';
                    });
                    
                    // Set up lazy loading for background images
                    setupLazyBackgrounds(element);
                }
            });
        });
        
        // Load third-party resources only after critical content
        loadThirdPartyResources();
    });
    
    // Apply background image priority
    function applyBackgroundImagePriority(element, priority) {
        const styles = window.getComputedStyle(element);
        const bgImage = styles.backgroundImage;
        
        if (bgImage && bgImage !== 'none') {
            // Preload important background images
            if (priority === 'high') {
                const url = bgImage.replace(/url\(['"]?(.*?)['"]?\)/i, '$1');
                if (url) {
                    const link = document.createElement('link');
                    link.rel = 'preload';
                    link.as = 'image';
                    link.href = url;
                    link.fetchpriority = 'high';
                    document.head.appendChild(link);
                }
            }
        }
    }
    
    // Setup lazy loading for background images
    function setupLazyBackgrounds(element) {
        const styles = window.getComputedStyle(element);
        const bgImage = styles.backgroundImage;
        
        if (bgImage && bgImage !== 'none') {
            // Save current background and replace with transparent
            element.dataset.background = bgImage.replace(/url\(['"]?(.*?)['"]?\)/i, '$1');
            element.style.backgroundImage = 'none';
            
            // Setup intersection observer for this element
            createObserver(element);
        }
        
        // Check child elements with background images
        element.querySelectorAll('[style*="background-image"]').forEach(child => {
            const childStyles = window.getComputedStyle(child);
            const childBgImage = childStyles.backgroundImage;
            
            if (childBgImage && childBgImage !== 'none') {
                child.dataset.background = childBgImage.replace(/url\(['"]?(.*?)['"]?\)/i, '$1');
                child.style.backgroundImage = 'none';
                
                createObserver(child);
            }
        });
    }
    
    // Create observer for background image lazy loading
    function createObserver(element) {
        // Use intersection observer API to load when visible
        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const target = entry.target;
                        if (target.dataset.background) {
                            target.style.backgroundImage = `url(${target.dataset.background})`;
                            delete target.dataset.background;
                        }
                        observer.unobserve(target);
                    }
                });
            }, { rootMargin: '200px' });
            
            observer.observe(element);
        } else {
            // Fallback for browsers without IntersectionObserver
            setTimeout(() => {
                if (element.dataset.background) {
                    element.style.backgroundImage = `url(${element.dataset.background})`;
                    delete element.dataset.background;
                }
            }, 1000);
        }
    }
    
    // Load third-party resources only after critical content
    function loadThirdPartyResources() {
        // Use requestIdleCallback to load non-critical third-party resources
        const loadNonCriticalResources = () => {
            // Analytics, tracking, social widgets, etc.
            document.querySelectorAll('script[data-priority="defer"]').forEach(script => {
                const newScript = document.createElement('script');
                Array.from(script.attributes).forEach(attr => {
                    if (attr.name !== 'data-priority') {
                        newScript.setAttribute(attr.name, attr.value);
                    }
                });
                newScript.async = true;
                newScript.defer = true;
                
                script.parentNode.replaceChild(newScript, script);
            });
        };
        
        // Use requestIdleCallback if available
        if ('requestIdleCallback' in window) {
            requestIdleCallback(loadNonCriticalResources, { timeout: 5000 });
        } else {
            // Fallback to setTimeout
            setTimeout(loadNonCriticalResources, 3000);
        }
    }
})();
