jQuery(function (jQuery) {
  "use strict";

  jQuery(window).on("load", function (e) {
    jQuery(".pq-masonry").isotope({
      itemSelector: ".pq-masonry-item",
      masonry: {
        columnWidth: ".grid-sizer",
        gutter: 0,
      },
    });

    jQuery(".pq-grid").isotope({
      itemSelector: ".pq-grid-item",
    });

    jQuery(".pq-filter-button-group").on(
      "click",
      ".pq-filter-btn",
      function () {
        var filterValue = jQuery(this).attr("data-filter");
        jQuery(".pq-masonry").isotope({ filter: filterValue });
        jQuery(".pq-grid").isotope({ filter: filterValue });
        jQuery(".pq-filter-button-group .pq-filter-btn").removeClass("active");
        jQuery(this).addClass("active");

        updateFilterCounts();
      }
    );

    var initial_items = 5;
    var next_items = 3;

    if (jQuery(".pq-masonry").length > 0) {
      var initial_items = jQuery(".pq-masonry").data("initial_items");
      var next_items = jQuery(".pq-masonry").data("next_items");
    }

    if (jQuery(".pq-grid").length > 0) {
      var initial_items = jQuery(".pq-grid").data("initial_items");
      var next_items = jQuery(".pq-grid").data("next_items");
    }

    function showNextItems(pagination) {
      var itemsMax = jQuery(".visible_item").length;
      var itemsCount = 0;
      jQuery(".visible_item").each(function () {
        if (itemsCount < pagination) {
          jQuery(this).removeClass("visible_item");
          itemsCount++;
        }
      });
      if (itemsCount >= itemsMax) {
        jQuery("#showMore").hide();
      }

      if (jQuery(".pq-masonry").length > 0) {
        jQuery(".pq-masonry").isotope("layout");
      }

      if (jQuery(".pq-grid").length > 0) {
        jQuery(".pq-grid").isotope("layout");
      }
    }
    // function that hides items when page is loaded - optimized for performance
    function hideItems(pagination) {
      var $items = jQuery(".pq-filter-items");
      var itemsMax = $items.length;
      var $showMoreButton = jQuery("#showMore");
      var hasVisibleItems = false;

      // Use cached jQuery references and faster DOM manipulation
      if (itemsMax > 0) {
        // Process all items in a single pass for better performance
        $items.each(function(index) {
          if (index >= pagination) {
            jQuery(this).addClass("visible_item");
            hasVisibleItems = true;
          }
        });

        // Only manipulate DOM when needed
        if (!hasVisibleItems || initial_items >= itemsMax) {
          $showMoreButton.hide();
        }

        // Defer layout recalculations to next frame for better performance
        requestAnimationFrame(function() {
          // Trigger layout only if elements exist to avoid unnecessary processing
          if (jQuery(".pq-masonry").length > 0) {
            jQuery(".pq-masonry").isotope("layout");
          }
          
          if (jQuery(".pq-grid").length > 0) {
            jQuery(".pq-grid").isotope("layout");
          }
        });
      } else {
        // No items to show, hide the button
        $showMoreButton.hide();
      }
    }
    jQuery("#showMore").on("click", function (e) {
      e.preventDefault();
      showNextItems(next_items);
    });
    hideItems(initial_items);

    function updateFilterCounts() {
      // get filtered item elements
      if (jQuery(".pq-masonry").length > 0) {
        var itemElems = jQuery(".pq-masonry").isotope(
          "getFilteredItemElements"
        );
      }
      if (jQuery(".pq-grid").length > 0) {
        var itemElems = jQuery(".pq-grid").isotope("getFilteredItemElements");
      }

      var count_items = jQuery(itemElems).length;

      if (count_items > initial_items) {
        jQuery("#showMore").show();
      } else {
        jQuery("#showMore").hide();
      }
      if (jQuery(".pq-filter-items").hasClass("visible_item")) {
        jQuery(".pq-filter-items").removeClass("visible_item");
      }
      var index = 0;

      jQuery(itemElems).each(function () {
        if (index >= initial_items) {
          jQuery(this).addClass("visible_item");
        }
        index++;
      });
      if (jQuery(".pq-masonry").length > 0) {
        jQuery(".pq-masonry").isotope("layout");
      }

      if (jQuery(".pq-grid").length > 0) {
        jQuery(".pq-grid").isotope("layout");
      }
    }
  });
});
