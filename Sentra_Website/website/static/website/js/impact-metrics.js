// Enhanced Impact Metrics JavaScript with advanced animations

document.addEventListener('DOMContentLoaded', function() {
    // Initialize impact metric animations
    initImpactMetrics();
    
    // Add hover effects to metric cards
    addMetricCardHoverEffects();
});

function initImpactMetrics() {
    // Add intersection observer for impact metrics animations
    const impactMetrics = document.querySelectorAll('.impact-metric-card');
    
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const impactObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add staggered animation delay
                const index = Array.from(impactMetrics).indexOf(entry.target);
                entry.target.style.animationDelay = `${index * 0.15}s`;
                
                // Add animated class to trigger CSS animation
                entry.target.classList.add('animated');
                
                // Get the counter element inside this card
                const counter = entry.target.querySelector('.pq-count');
                if (counter) {
                    const targetValue = parseFloat(counter.getAttribute('data-count'));
                    const duration = parseInt(counter.getAttribute('data-pq-duration')) || 2000;
                    
                    // Add slight delay before starting counter animation
                    setTimeout(() => {
                        // Animate the counter when it comes into view
                        animateCounterOnScroll(counter, targetValue, duration);
                    }, index * 150 + 300);
                }
                
                // Stop observing after animation
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Start observing each impact metric card
    impactMetrics.forEach(metric => {
        impactObserver.observe(metric);
    });
}

function addMetricCardHoverEffects() {
    const cards = document.querySelectorAll('.impact-metric-card');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            // Add a pulse animation to the icon
            const icon = this.querySelector('.impact-metric-icon');
            if (icon) {
                icon.style.animation = 'pulse 1s infinite';
            }
            
            // Highlight this card and dim siblings
            cards.forEach(sibling => {
                if (sibling !== card) {
                    sibling.style.opacity = '0.7';
                    sibling.style.transform = 'scale(0.98)';
                }
            });
        });
        
        card.addEventListener('mouseleave', function() {
            // Remove pulse animation from icon
            const icon = this.querySelector('.impact-metric-icon');
            if (icon) {
                icon.style.animation = '';
            }
            
            // Reset all cards
            cards.forEach(sibling => {
                sibling.style.opacity = '1';
                sibling.style.transform = '';
            });
        });
    });
}

// Function to animate counter on scroll
function animateCounterOnScroll(counterElement, targetValue, duration) {
    let startTime = null;
    let currentValue = 0;
    
    function updateCounter(timestamp) {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1);
        
        // Using easeOutExpo for smoother animation
        const easedProgress = 1 - Math.pow(1 - progress, 3);
        
        // Check if target is a number with decimal point
        if (targetValue % 1 !== 0) {
            // Handle decimal numbers with 1 decimal place
            currentValue = (targetValue * easedProgress).toFixed(1);
            counterElement.textContent = currentValue;
        } else {
            // Handle integer numbers
            currentValue = Math.floor(targetValue * easedProgress);
            counterElement.textContent = currentValue;
        }
        
        // Continue animation until complete
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            // Ensure final value is exactly the target
            counterElement.textContent = targetValue;
            
            // Add a small bounce effect at the end
            counterElement.style.transform = 'scale(1.1)';
            setTimeout(() => {
                counterElement.style.transform = 'scale(1)';
            }, 100);
        }
    }
    
    // Start the counter animation
    requestAnimationFrame(updateCounter);
}

// Add a pulse animation keyframe
if (!document.getElementById('impact-metrics-keyframes')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'impact-metrics-keyframes';
    styleSheet.textContent = `
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.08); }
            100% { transform: scale(1); }
        }
    `;
    document.head.appendChild(styleSheet);
}

// Add parallax effect to decoration elements
document.addEventListener('DOMContentLoaded', function() {
    const decorations = document.querySelectorAll('.impact-metrics-decoration');
    
    window.addEventListener('scroll', function() {
        const scrollPosition = window.scrollY;
        
        decorations.forEach((decoration, index) => {
            const speed = 0.05 * (index + 1);
            decoration.style.transform = `translate3d(0, ${scrollPosition * speed}px, 0)`;
        });
    });
});
