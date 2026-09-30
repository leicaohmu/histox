/* Add direct actions beside executable example summaries. */
(function () {
  "use strict";

  function buildExampleActions() {
    document.querySelectorAll("[data-histox-example-actions]").forEach(function (toolbar) {
      if (toolbar.dataset.ready === "true") {
        return;
      }

      var githubLink = toolbar.querySelector(".histox-example-action--github");
      var downloads = [
        [".sphx-glr-download-jupyter a", "Download notebook"],
        [".sphx-glr-download-python a", "Download Python"],
      ];

      downloads.forEach(function (download) {
        var sourceLink = document.querySelector(download[0]);
        if (!sourceLink) {
          return;
        }

        var action = sourceLink.cloneNode(false);
        action.className = "histox-example-action";
        action.textContent = download[1];
        toolbar.insertBefore(action, githubLink);
      });

      toolbar.dataset.ready = "true";
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildExampleActions);
  } else {
    buildExampleActions();
  }
})();
