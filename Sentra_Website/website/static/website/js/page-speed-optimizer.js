/**
 * Advanced Page Speed Optimizer Script
 * Implements modern resource prioritization, LCP optimization, and progressive loading
 * Reduces Time-to-Interactive and optimizes Core Web Vitals
 */

// Continue execution if we started performance tracking in the page header
(function() {
    // Initialize or continue performance tracking
    if (!window.perfData) window.perfData = { start: performance.now() };
    
    // Track key metrics for Core Web Vitals
    const vitalsData = {
        firstPaintTime: 0,
        lcpTime: 0,
        fidTime: 0,
        clsValue: 0
    };
    
    // Register early event listeners for Core Web Vitals
    const po = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
            if (entry.name === 'first-paint') {
                vitalsData.firstPaintTime = entry.startTime;
            }
            if (entry.name === 'largest-contentful-paint') {
                vitalsData.lcpTime = entry.startTime;
            }
        }
    });
    po.observe({ type: 'paint', buffered: true });
    
    // Priority-based resource loading strategy
    function optimizeResourceLoading() {
        // Phase 1: Handle critical above-the-fold resources
        const loadingElement = document.getElementById('pq-loading');
        
        // Optimize LCP element (usually the hero image)
        const lcpCandidates = document.querySelectorAll('.smartel-slider img, .banner img, header img');
        lcpCandidates.forEach(img => {
            if (!img.loading) img.loading = 'eager'; // Force eager loading for LCP candidates
            if (!img.fetchpriority) img.fetchpriority = 'high'; // Set high priority
            img.dataset.priority = 'critical'; // Mark as critical
        });
        
        // Phase 2: Setup lazy loading for below-the-fold content
        setupProgressiveLoading();
        
        // Phase 3: Handle loading indicator
        if (loadingElement) {
            // Hide loading animation after critical content is ready
            const hideLoader = () => {
                if (loadingElement.style.display !== 'none') {
                    loadingElement.style.transition = 'opacity 0.3s ease';
                    loadingElement.style.opacity = '0';
                    setTimeout(() => { loadingElement.style.display = 'none'; }, 300);
                }
            };
            
            // Use requestIdleCallback for non-critical UI updates
            if ('requestIdleCallback' in window) {
                requestIdleCallback(() => {
                    if (document.readyState === 'complete') hideLoader();
                    else window.addEventListener('load', hideLoader);
                }, { timeout: 2000 });
            } else {
                if (document.readyState === 'complete') setTimeout(hideLoader, 500);
                else window.addEventListener('load', () => setTimeout(hideLoader, 500));
            }
            
            // Safety timeout - hide loader even if something fails to load
            setTimeout(() => hideLoader(), 3000);
        }
    }
    
    // Setup progressive loading for non-critical resources
    function setupProgressiveLoading() {
        // Prioritize important elements first
        const priorityElements = [
            { selector: 'header, nav, .navbar, .pq-header-style-1', priority: 'high' },
            { selector: '.banner, .hero-section, .smartel-slider', priority: 'high' },
            { selector: '#about, .about-section', priority: 'medium' },
            { selector: 'footer', priority: 'low' }
        ];
        
        // Set priority attributes on key DOM elements
        priorityElements.forEach(item => {
            document.querySelectorAll(item.selector).forEach(el => {
                el.dataset.loadPriority = item.priority;
            });
        });
        
        // Use Intersection Observer for smart lazy loading
        const observerOptions = {
            rootMargin: '200px', // Load before element comes into view
            threshold: 0
        };
        
        const lazyLoadObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const target = entry.target;
                    
                    // Handle images
                    if (target.tagName === 'IMG') {
                        if (target.dataset.src) {
                            target.src = target.dataset.src;
                            target.removeAttribute('data-src');
                        }
                        target.dataset.lazyLoaded = "loaded";
                    }
                    
                    // Handle background images
                    if (target.dataset.background) {
                        target.style.backgroundImage = `url('${target.dataset.background}')`;
                        target.removeAttribute('data-background');
                    }
                    
                    lazyLoadObserver.unobserve(target);
                }
            });
        }, observerOptions);
        
        // Apply lazy loading to images below the fold
        document.querySelectorAll('img:not([data-priority="critical"]):not([loading="eager"])').forEach(img => {
            // Don't process already handled images
            if (img.dataset.lazyLoaded) return;
            
            // Check if image is below the fold
            if (!isElementInViewport(img) && !img.closest('#pq-loading')) {
                img.dataset.src = img.src;
                img.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'%3E%3C/svg%3E";
                img.dataset.lazyLoaded = "pending";
                img.loading = "lazy"; // Use native lazy loading as backup
                lazyLoadObserver.observe(img);
            }
        });
        
        // Apply lazy loading to elements with background images
        document.querySelectorAll('[data-background]').forEach(el => {
            lazyLoadObserver.observe(el);
        });
    }

    // Utility: Check if element is visible in viewport
    function isElementInViewport(el) {
        const rect = el.getBoundingClientRect();
        return (
            rect.top <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.top + rect.height >= 0
        );
    }
    
    // Execute optimization when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', optimizeResourceLoading);
    } else {
        optimizeResourceLoading();
    }
    
    // Track performance and report when everything is done
    window.addEventListener('load', function() {
        window.perfData.windowLoaded = performance.now();
        
        // Allow layout to settle before final calculations
        setTimeout(() => {
            // Calculate key performance metrics
            const totalLoadTime = window.perfData.windowLoaded - window.perfData.start;
            const firstPaint = vitalsData.firstPaintTime || 0;
            const lcp = vitalsData.lcpTime || 0;
            
            // Log performance data for analysis
            console.log('Performance metrics:', {
                totalLoadTime: `${Math.round(totalLoadTime)}ms`,
                firstPaint: firstPaint ? `${Math.round(firstPaint)}ms` : 'Not available',
                largestContentfulPaint: lcp ? `${Math.round(lcp)}ms` : 'Not available'
            });
            
            // Clean up observers
            po.disconnect();
        }, 0);
    });
});
