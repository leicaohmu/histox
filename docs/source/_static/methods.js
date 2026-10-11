(() => {
  "use strict";

  const initializeRegistry = () => {
    const registry = document.querySelector("[data-method-registry]");
    if (!registry) return;

    const entries = Array.from(registry.querySelectorAll("[data-method-entry]"));
    const filters = Array.from(registry.querySelectorAll("[data-method-filter]"));
    const search = registry.querySelector("[data-method-search]");
    const results = registry.querySelector("[data-method-results]");
    const empty = registry.querySelector("[data-method-empty]");
    let activeTask = "all";

    const normalize = (value) => value.trim().toLocaleLowerCase();

    const update = () => {
      const query = normalize(search.value);
      let visibleCount = 0;

      entries.forEach((entry) => {
        const matchesTask = activeTask === "all" || entry.dataset.task === activeTask;
        const matchesQuery = !query || normalize(entry.textContent).includes(query);
        const isVisible = matchesTask && matchesQuery;
        entry.hidden = !isVisible;
        if (isVisible) visibleCount += 1;
      });

      results.textContent = `${visibleCount} ${visibleCount === 1 ? "method" : "methods"} shown`;
      empty.hidden = visibleCount !== 0;
    };

    filters.forEach((filter) => {
      filter.addEventListener("click", () => {
        activeTask = filter.dataset.methodFilter;
        filters.forEach((candidate) => {
          const isActive = candidate === filter;
          candidate.classList.toggle("is-active", isActive);
          candidate.setAttribute("aria-pressed", String(isActive));
        });
        update();
      });
    });

    search.addEventListener("input", update);
    filters[0].classList.add("is-active");
    update();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeRegistry, { once: true });
  } else {
    initializeRegistry();
  }
})();
