(function () {
  "use strict";

  function initGlobalMenus() {
    var menus = Array.from(document.querySelectorAll(".histox-nav-menu"));

    function closeMenu(menu, restoreFocus) {
      var trigger = menu.querySelector(".histox-nav-menu__trigger");
      menu.classList.remove("is-open");
      if (menu.tagName === "DETAILS") menu.open = false;
      if (trigger) {
        trigger.setAttribute("aria-expanded", "false");
        if (restoreFocus) trigger.focus();
      }
    }

    function closeAll(exceptMenu) {
      menus.forEach(function (menu) {
        if (menu !== exceptMenu) closeMenu(menu, false);
      });
    }

    menus.forEach(function (menu) {
      var trigger = menu.querySelector(".histox-nav-menu__trigger");
      if (!trigger) return;

      if (menu.tagName === "DETAILS") {
        menu.addEventListener("toggle", function () {
          menu.classList.toggle("is-open", menu.open);
          trigger.setAttribute("aria-expanded", String(menu.open));
          if (menu.open) closeAll(menu);
        });
        return;
      }

      trigger.addEventListener("click", function () {
        var willOpen = !menu.classList.contains("is-open");
        closeAll(menu);
        menu.classList.toggle("is-open", willOpen);
        trigger.setAttribute("aria-expanded", String(willOpen));
      });

      menu.querySelectorAll("a").forEach(function (link) {
        link.addEventListener("click", function () {
          closeMenu(menu, false);
        });
      });
    });

    document.addEventListener("click", function (event) {
      if (event.target.closest(".histox-nav-menu")) return;
      closeAll(null);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") return;
      var openMenu = document.querySelector(".histox-nav-menu.is-open");
      if (openMenu) closeMenu(openMenu, true);
    });
  }

  function initTutorialMenus() {
    var menus = Array.from(document.querySelectorAll(".histox-tutorial-category"));

    menus.forEach(function (menu) {
      var toggle = menu.querySelector(".histox-tutorial-category__toggle");
      if (!toggle) return;

      toggle.addEventListener("click", function () {
        var willOpen = !menu.classList.contains("is-open");
        menus.forEach(function (otherMenu) {
          otherMenu.classList.remove("is-open");
          var otherToggle = otherMenu.querySelector(".histox-tutorial-category__toggle");
          if (otherToggle) otherToggle.setAttribute("aria-expanded", "false");
        });
        menu.classList.toggle("is-open", willOpen);
        toggle.setAttribute("aria-expanded", String(willOpen));
      });

      menu.querySelectorAll("a").forEach(function (link) {
        link.addEventListener("click", function () {
          menu.classList.remove("is-open");
          toggle.setAttribute("aria-expanded", "false");
        });
      });
    });

    document.addEventListener("click", function (event) {
      if (event.target.closest(".histox-tutorial-category")) return;
      menus.forEach(function (menu) {
        menu.classList.remove("is-open");
        var toggle = menu.querySelector(".histox-tutorial-category__toggle");
        if (toggle) toggle.setAttribute("aria-expanded", "false");
      });
    });

    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") return;
      menus.forEach(function (menu) {
        menu.classList.remove("is-open");
        var toggle = menu.querySelector(".histox-tutorial-category__toggle");
        if (toggle) toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  function initTutorialFilters() {
    var library = document.querySelector("[data-histox-tutorial-library]");
    if (!library) return;

    var buttons = Array.from(library.querySelectorAll("[data-topic]"));
    var cards = Array.from(library.querySelectorAll("[data-topics]"));
    var resultCount = library.querySelector("[data-result-count]");
    var knownTopics = buttons.map(function (button) {
      return button.dataset.topic;
    });

    function topicFromHash() {
      var match = window.location.hash.match(/^#topic-([a-z-]+)$/);
      return match && knownTopics.indexOf(match[1]) !== -1 ? match[1] : null;
    }

    function showTopic(topic, updateHash) {
      var selected = knownTopics.indexOf(topic) !== -1 ? topic : "all";
      var visible = 0;

      buttons.forEach(function (button) {
        var isSelected = button.dataset.topic === selected;
        button.classList.toggle("is-active", isSelected);
        button.setAttribute("aria-pressed", String(isSelected));
      });

      cards.forEach(function (card) {
        var topics = card.dataset.topics.split(/\s+/);
        var matches = selected === "all" || topics.indexOf(selected) !== -1;
        card.hidden = !matches;
        if (matches) visible += 1;
      });

      if (resultCount) resultCount.textContent = String(visible);

      if (updateHash) {
        var nextHash = selected === "all" ? "#tutorial-library" : "#topic-" + selected;
        window.history.replaceState(null, "", nextHash);
      }
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        showTopic(button.dataset.topic, true);
      });
    });

    document.querySelectorAll('a[href^="#topic-"]').forEach(function (link) {
      link.addEventListener("click", function () {
        var topic = link.getAttribute("href").replace("#topic-", "");
        showTopic(topic, false);
      });
    });

    window.addEventListener("hashchange", function () {
      var topic = topicFromHash();
      if (topic) showTopic(topic, false);
    });

    showTopic(topicFromHash() || "all", false);
  }

  function initTutorials() {
    initGlobalMenus();
    initTutorialMenus();
    initTutorialFilters();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTutorials);
  } else {
    initTutorials();
  }
})();
