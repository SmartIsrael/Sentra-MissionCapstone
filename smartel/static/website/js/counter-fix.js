/**
 * Enhanced counter animation for Smartel website
 * This script ensures counters animate properly on scroll
 */

// Immediately invoked function to avoid global scope pollution
(function($) {
  "use strict";
  
  // Function to animate a single counter
  function animateCounter($counter) {
    // Skip if already animated
    if ($counter.hasClass('counted')) return;
    
    var countTo = $counter.attr('data-count');
    var duration = parseInt($counter.attr('data-pq-duration')) || 2000;
    
    // Mark as counted to prevent re-animation
    $counter.addClass('counted');
    
    // Reset to zero first
    $counter.text('0');
    
    // Animate the counter
    $({ Counter: 0 }).animate({
      Counter: countTo
    }, {
      duration: duration,
      easing: 'swing',
      step: function() {
        // For decimal values, show one decimal place
        if (this.Counter % 1 !== 0) {
          $counter.text(this.Counter.toFixed(1));
        } else {
          $counter.text(Math.floor(this.Counter));
        }
      },
      complete: function() {
        // Ensure final value is exact
        $counter.text(countTo);
      }
    });
  }
  
  // Function to animate all counters that are not yet counted
  function animateCounters() {
    $('.pq-count').each(function() {
      animateCounter($(this));
    });
  }

  // Function to check if element is in viewport with a threshold
  function isElementInViewport(el) {
    if (typeof jQuery === "function" && el instanceof jQuery) {
      el = el[0];
    }
    
    var rect = el.getBoundingClientRect();
    var windowHeight = window.innerHeight || document.documentElement.clientHeight;
    
    // Only trigger animation when element is at least 20% visible in the viewport
    var visibleThreshold = 0.2;
    var visibleHeight = Math.min(rect.bottom, windowHeight) - Math.max(rect.top, 0);
    var elementHeight = rect.bottom - rect.top;
    var visibleRatio = visibleHeight / elementHeight;
    
    return visibleRatio > visibleThreshold && 
           rect.top < windowHeight * 0.8 && 
           rect.bottom > windowHeight * 0.2;
  }

  // Check all counters to see if they're visible and should be animated
  function checkCounters() {
    $('.pq-count').each(function() {
      var $this = $(this);
      
      // If counter is in viewport and not yet animated, animate it
      if (isElementInViewport($this) && !$this.hasClass('counted')) {
        animateCounter($this);
      }
    });
  }
  
  // Initialize on document ready
  $(document).ready(function() {
    // Don't animate on page load, wait until scroll
    
    // Check counters on scroll
    $(window).on('scroll', function() {
      checkCounters();
    });
    
    // Force an initial check after page is fully loaded
    $(window).on('load', function() {
      setTimeout(checkCounters, 500);
    });
    
    // Manually trigger for specific sections on hover
    $('.pq-counter').parent().on('mouseenter', function() {
      $(this).find('.pq-count').each(function() {
        var $this = $(this);
        if (!$this.hasClass('counted')) {
          animateCounter($this);
        }
      });
    });
  });
  
  
  // Trigger a scroll check now, in case elements are already visible
  checkCounters();
  
  // Trigger once more after a delay to ensure everything's properly rendered
  setTimeout(checkCounters, 1000);
  
})(jQuery);
