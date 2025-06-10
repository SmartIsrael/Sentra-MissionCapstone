// page-1 title
var tpj = jQuery;
if (window.RS_MODULES === undefined) window.RS_MODULES = {};
if (RS_MODULES.modules === undefined) RS_MODULES.modules = {};
RS_MODULES.modules["revslider81"] = {
  once:
    RS_MODULES.modules["revslider81"] !== undefined
      ? RS_MODULES.modules["revslider81"].once
      : undefined,
  init: function () {
    window.revapi8 =
      window.revapi8 === undefined ||
      window.revapi8 === null ||
      window.revapi8.length === 0
        ? document.getElementById("rev_slider_8_1")
        : window.revapi8;
    if (
      window.revapi8 === null ||
      window.revapi8 === undefined ||
      window.revapi8.length == 0
    ) {
      window.revapi8initTry =
        window.revapi8initTry === undefined ? 0 : window.revapi8initTry + 1;
      if (window.revapi8initTry < 20)
        requestAnimationFrame(function () {
          RS_MODULES.modules["revslider81"].init();
        });
      return;
    }
    window.revapi8 = jQuery(window.revapi8);
    if (window.revapi8.revolution == undefined) {
      revslider_showDoubleJqueryError("rev_slider_8_1");
      return;
    }
    revapi8.revolutionInit({
      revapi: "revapi8",
      DPR: "dpr",
      sliderLayout: "fullwidth",
      visibilityLevels: "1240,1024,778,480",
      gridwidth: "1300,1024,778,480",
      gridheight: "980,700,800,600",
      perspective: 600,
      perspectiveType: "global",
      editorheight: "980,700,800,600",
      responsiveLevels: "1240,1024,778,480",
      progressBar: { disableProgressBar: true },
      navigation: {
        wheelCallDelay: 1000,
        onHoverStop: false,
        arrows: {
          enable: true,
          tmp: '<div class="tp-title-wrap">  	<div class="tp-arr-imgholder"></div> </div>',
          style: "zeus",
          hide_onmobile: true,
          hide_under: "1023px",
          left: {
            anim: "zoomin",
            h_offset: 30,
            v_offset: 70,
          },
          right: {
            h_offset: 30,
            v_offset: 70,
          },
        },
        bullets: {
          enable: true,
          tmp: "",
          style: "hermes",
          hide_over: "1023px",
        },
      },
      viewPort: {
        global: false,
        globalDist: "-200px",
        enable: false,
      },
      fallbacks: {
        allowHTML5AutoPlayOnAndroid: true,
      },
    });
  },
}; // End of RevInitScript
if (window.RS_MODULES.checkMinimal !== undefined) {
  window.RS_MODULES.checkMinimal();
}

// page-2 title
var tpj = jQuery;
if (window.RS_MODULES === undefined) window.RS_MODULES = {};
if (RS_MODULES.modules === undefined) RS_MODULES.modules = {};
RS_MODULES.modules["revslider91"] = {
  once:
    RS_MODULES.modules["revslider91"] !== undefined
      ? RS_MODULES.modules["revslider91"].once
      : undefined,
  init: function () {
    window.revapi9 =
      window.revapi9 === undefined ||
      window.revapi9 === null ||
      window.revapi9.length === 0
        ? document.getElementById("rev_slider_9_1")
        : window.revapi9;
    if (
      window.revapi9 === null ||
      window.revapi9 === undefined ||
      window.revapi9.length == 0
    ) {
      window.revapi9initTry =
        window.revapi9initTry === undefined ? 0 : window.revapi9initTry + 1;
      if (window.revapi9initTry < 20)
        requestAnimationFrame(function () {
          RS_MODULES.modules["revslider91"].init();
        });
      return;
    }
    window.revapi9 = jQuery(window.revapi9);
    if (window.revapi9.revolution == undefined) {
      revslider_showDoubleJqueryError("rev_slider_9_1");
      return;
    }
    revapi9.revolutionInit({
      revapi: "revapi9",
      DPR: "dpr",
      sliderLayout: "fullwidth",
      visibilityLevels: "1240,1024,778,480",
      gridwidth: "1300,1024,778,480",
      gridheight: "900,710,1000,780",
      perspective: 600,
      perspectiveType: "global",
      editorheight: "900,710,1000,780",
      responsiveLevels: "1240,1024,778,480",
      progressBar: { disableProgressBar: true },
      navigation: {
        wheelCallDelay: 1000,
        onHoverStop: false,
        arrows: {
          enable: true,
          tmp: '<div class="tp-title-wrap">  	<div class="tp-arr-imgholder"></div> </div>',
          style: "zeus",
          hide_onmobile: true,
          hide_under: "799px",
          left: {
            anim: "right",
            container: "layergrid",
            v_align: "bottom",
            h_offset: 30,
            v_offset: 90,
          },
          right: {
            anim: "right",
            container: "layergrid",
            h_align: "left",
            v_align: "bottom",
            h_offset: 110,
            v_offset: 90,
          },
        },
        bullets: {
          enable: true,
          tmp: "",
          style: "hermes",
          hide_over: "799px",
          v_offset: 30,
        },
      },
      viewPort: {
        global: false,
        globalDist: "-200px",
        enable: false,
      },
      fallbacks: {
        allowHTML5AutoPlayOnAndroid: true,
      },
    });
  },
}; // End of RevInitScript
if (window.RS_MODULES.checkMinimal !== undefined) {
  window.RS_MODULES.checkMinimal();
}

// page-3 title
var tpj = jQuery;
if (window.RS_MODULES === undefined) window.RS_MODULES = {};
if (RS_MODULES.modules === undefined) RS_MODULES.modules = {};
RS_MODULES.modules["revslider111"] = {
  once:
    RS_MODULES.modules["revslider111"] !== undefined
      ? RS_MODULES.modules["revslider111"].once
      : undefined,
  init: function () {
    window.revapi11 =
      window.revapi11 === undefined ||
      window.revapi11 === null ||
      window.revapi11.length === 0
        ? document.getElementById("rev_slider_11_1")
        : window.revapi11;
    if (
      window.revapi11 === null ||
      window.revapi11 === undefined ||
      window.revapi11.length == 0
    ) {
      window.revapi11initTry =
        window.revapi11initTry === undefined ? 0 : window.revapi11initTry + 1;
      if (window.revapi11initTry < 20)
        requestAnimationFrame(function () {
          RS_MODULES.modules["revslider111"].init();
        });
      return;
    }
    window.revapi11 = jQuery(window.revapi11);
    if (window.revapi11.revolution == undefined) {
      revslider_showDoubleJqueryError("rev_slider_11_1");
      return;
    }
    revapi11.revolutionInit({
      revapi: "revapi11",
      DPR: "dpr",
      sliderLayout: "fullwidth",
      visibilityLevels: "1240,1024,778,480",
      gridwidth: "1300,1024,778,480",
      gridheight: "900,700,700,550",
      perspective: 600,
      perspectiveType: "global",
      editorheight: "900,700,700,550",
      responsiveLevels: "1240,1024,778,480",
      progressBar: { disableProgressBar: true },
      navigation: {
        wheelCallDelay: 1000,
        onHoverStop: false,
        arrows: {
          enable: true,
          tmp: '<div class="tp-title-wrap">  	<div class="tp-arr-imgholder"></div> </div>',
          style: "zeus",
          hide_onmobile: true,
          hide_under: "799px",
          left: {
            anim: "zoomin",
            container: "layergrid",
            h_align: "right",
            h_offset: 0,
            v_offset: 10,
          },
          right: {
            anim: "zoomin",
            container: "layergrid",
            h_offset: 0,
            v_offset: 100,
          },
        },
        bullets: {
          enable: true,
          tmp: "",
          style: "hermes",
          hide_over: "799px",
          v_offset: 30,
        },
      },
      viewPort: {
        global: false,
        globalDist: "-200px",
        enable: false,
      },
      fallbacks: {
        allowHTML5AutoPlayOnAndroid: true,
      },
    });
  },
};
