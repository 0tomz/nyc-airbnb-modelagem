// Reveal the EDA appendix when following a link to one of its sections.
(() => {
  function reveal(hash) {
    if (!hash || hash === "#") return null;
    let id;
    try { id = decodeURIComponent(hash.slice(1)); } catch { return null; }
    const target = document.getElementById(id);
    if (!target) return null;
    let disclosure = target.closest("details");
    while (disclosure) {
      disclosure.open = true;
      disclosure = disclosure.parentElement?.closest("details");
    }
    return target;
  }
  document.addEventListener("click", event => {
    const anchor = event.target.closest("a[href^='#']");
    if (anchor) reveal(anchor.getAttribute("href"));
  });
  window.addEventListener("hashchange", () => reveal(window.location.hash));
  const initial = reveal(window.location.hash);
  if (initial) requestAnimationFrame(() => initial.scrollIntoView());
})();
