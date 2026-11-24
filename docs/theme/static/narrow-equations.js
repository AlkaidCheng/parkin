/**
 * Adapts display equations to a narrow screen.
 *
 * The pages write their equations for the full text column. Two macros mark
 * where an equation may be broken: book.yaml defines \nb as nothing and \ns
 * as a wide space, and on a screen that matches NARROW_QUERY this script
 * turns both into a new row of the `aligned` environment they stand in. An
 * equation that is still wider than its column is then drawn smaller, down to
 * MIN_SCALE of its size; below that it scrolls, as every equation would
 * without this script.
 *
 * A page typeset for a wide screen is loaded again when the screen becomes
 * narrow.
 *
 * The script must run after the MathJax configuration in the page head and
 * before MathJax itself, which loads deferred.
 */
(function () {
  "use strict";

  // A phone held upright. At this width the text column is under half of the
  // 47rem (--measure in bookspine's book.css) that the equations are written for.
  var NARROW_QUERY = "(max-width: 37.5rem)";

  // The smallest an equation is drawn, as a fraction of its own size. At 0.7
  // the subscripts of an equation in 17px text are still about 8px tall.
  var MIN_SCALE = 0.7;

  // A hair under the exact ratio, so that rounding leaves no pixel to scroll.
  var FIT_MARGIN = 0.995;

  // How long the window must hold a size before the equations are fitted again.
  var RESIZE_DELAY_MS = 150;

  var config = window.MathJax;
  if (!config || !config.tex) {
    return;
  }
  var narrowScreen = window.matchMedia(NARROW_QUERY);
  var isTypesetNarrow = narrowScreen.matches;
  if (isTypesetNarrow) {
    config.tex.macros = config.tex.macros || {};
    // A new row, indented, with an empty term so that a sign that follows is
    // spaced as in a sum.
    config.tex.macros.nb = "\\\\ & \\quad {}";
    // A new row for the next of several formulas in one display.
    config.tex.macros.ns = "\\\\ &";
  }

  function fitEquations() {
    var boxes = document.querySelectorAll("div.arithmatex");
    for (var index = 0; index < boxes.length; index++) {
      var box = boxes[index];
      var math = box.querySelector("mjx-container");
      if (!math) {
        continue;
      }
      // MathJax sets a font size of its own on the container; keep it to
      // return to before each measurement.
      if (math.dataset.ownFontSize === undefined) {
        math.dataset.ownFontSize = math.style.fontSize;
      }
      math.style.fontSize = math.dataset.ownFontSize;
      if (box.scrollWidth <= box.clientWidth) {
        continue;
      }
      var scale = Math.max(MIN_SCALE, (box.clientWidth / box.scrollWidth) * FIT_MARGIN);
      var ownSize = parseFloat(window.getComputedStyle(math).fontSize);
      math.style.fontSize = (ownSize * scale).toFixed(2) + "px";
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    // A page with no math does not load MathJax.
    if (!window.MathJax || !window.MathJax.startup) {
      return;
    }
    window.MathJax.startup.promise.then(fitEquations);
    var timer = null;
    window.addEventListener("resize", function () {
      window.clearTimeout(timer);
      timer = window.setTimeout(fitEquations, RESIZE_DELAY_MS);
    });

    // MathJax reads the macros once. A page typeset for a wide screen is
    // therefore loaded again when the screen becomes narrow, as when a phone
    // is turned upright. The other way round the rows still fit, so the page
    // stays as it is.
    function reloadWhenNarrow(change) {
      if (change.matches && !isTypesetNarrow) {
        window.location.reload();
      }
    }
    if (narrowScreen.addEventListener) {
      narrowScreen.addEventListener("change", reloadWhenNarrow);
    } else if (narrowScreen.addListener) {
      narrowScreen.addListener(reloadWhenNarrow);
    }
  });
})();
