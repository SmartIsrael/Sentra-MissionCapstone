/**
 * Responsive Cards JS - Mobile-optimized card interactions
 * Enhances touch interactions and responsiveness for service cards
 */
document.addEventListener('DOMContentLoaded', function() {
    initResponsiveCards();
    
    // Re-initialize on window resize
    window.addEventListener('resize', debounce(function() {
        initResponsiveCards();
    }, 250));
});

/**
 * Initialize responsive card behaviors based on screen size
 */
function initResponsiveCards() {
    const serviceCards = document.querySelectorAll('.pq-service-card');
    const isMobile = window.matchMedia('(max-width: 767px)').matches;
    const isSmallMobile = window.matchMedia('(max-width: 480px)').matches;
    
    serviceCards.forEach((card, index) => {
        // Remove existing event listeners (to prevent duplicates)
        card.replaceWith(card.cloneNode(true));
        
        // Get the fresh reference after cloning
        const freshCard = document.querySelectorAll('.pq-service-card')[index];
        const cardInner = freshCard.querySelector('.card-inner');
        
        // Add staggered entrance animations on page load
        setTimeout(() => {
            freshCard.classList.add('animated-in');
        }, index * (isMobile ? 100 : 150));
        
        if (isMobile) {
            // For mobile: Improved touch interactions
            freshCard.addEventListener('touchstart', function(e) {
                // Track touch start position
                this.touchStartX = e.touches[0].clientX;
                this.touchStartY = e.touches[0].clientY;
            }, {passive: true});
            
            freshCard.addEventListener('touchend', function(e) {
                // Simple tap detection (no significant movement)
                const touchEndX = e.changedTouches[0].clientX;
                const touchEndY = e.changedTouches[0].clientY;
                const deltaX = Math.abs(touchEndX - this.touchStartX);
                const deltaY = Math.abs(touchEndY - this.touchStartY);
                
                // If it's a simple tap (not a scroll attempt)
                if (deltaX < 10 && deltaY < 10) {
                    e.preventDefault();
                    
                    // Toggle flip state
                    const currentTransform = cardInner.style.transform;
                    if (currentTransform === 'rotateY(180deg)') {
                        cardInner.style.transform = 'rotateY(0deg)';
                        
                        // Reset siblings
                        serviceCards.forEach(sibling => {
                            const siblingInner = sibling.querySelector('.card-inner');
                            if (sibling !== freshCard && siblingInner) {
                                siblingInner.style.transform = 'rotateY(0deg)';
                            }
                        });
                    } else {
                        cardInner.style.transform = 'rotateY(180deg)';
                        
                        // Reset other cards
                        serviceCards.forEach(sibling => {
                            const siblingInner = sibling.querySelector('.card-inner');
                            if (sibling !== freshCard && siblingInner) {
                                siblingInner.style.transform = 'rotateY(0deg)';
                            }
                        });
                    }
                }
            });
        } else {
            // Desktop: Enhanced hover effects
            freshCard.addEventListener('mouseenter', function() {
                cardInner.style.transform = 'rotateY(180deg)';
                
                // Dim siblings for focus effect
                serviceCards.forEach(sibling => {
                    if (sibling !== freshCard) {
                        sibling.style.opacity = '0.8';
                        sibling.style.transform = 'scale(0.98)';
                    }
                });
            });
            
            freshCard.addEventListener('mouseleave', function() {
                cardInner.style.transform = 'rotateY(0deg)';
                
                // Reset siblings
                serviceCards.forEach(sibling => {
                    sibling.style.opacity = '1';
                    sibling.style.transform = 'scale(1)';
                });
            });
        }
    });
    
    // Optimize the height for small mobile screens
    if (isSmallMobile) {
        optimizeCardsForSmallScreens();
    }
}

/**
 * Further optimize cards for very small screens
 */
function optimizeCardsForSmallScreens() {
    const descriptions = document.querySelectorAll('.pq-service-box-description');
    
    descriptions.forEach(desc => {
        // Truncate description text if too long for tiny screens
        if (desc.textContent.length > 120) {
            const originalText = desc.textContent;
            const truncatedText = originalText.substring(0, 117) + '...';
            
            // Save original text as data attribute
            desc.setAttribute('data-full-text', originalText);
            desc.textContent = truncatedText;
            
            // Add indicator that there's more text
            const moreIndicator = document.createElement('span');
            moreIndicator.classList.add('more-indicator');
            moreIndicator.textContent = 'Tap for more';
            moreIndicator.style.display = 'block';
            moreIndicator.style.fontSize = '12px';
            moreIndicator.style.fontStyle = 'italic';
            moreIndicator.style.marginTop = '5px';
            moreIndicator.style.color = 'rgba(255,255,255,0.7)';
            
            desc.parentNode.appendChild(moreIndicator);
            
            // Toggle between short and full text on tap
            desc.parentNode.addEventListener('click', function(e) {
                e.stopPropagation();
                if (desc.textContent === truncatedText) {
                    desc.textContent = desc.getAttribute('data-full-text');
                    moreIndicator.textContent = 'Tap to collapse';
                } else {
                    desc.textContent = truncatedText;
                    moreIndicator.textContent = 'Tap for more';
                }
            });
        }
    });
}

/**
 * Debounce function to limit expensive operations
 */
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            func.apply(context, args);
        }, wait);
    };
}
