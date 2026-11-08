const library = document.querySelector("#refmod-library");

function label(token) {
    return token?.querySelector("span")?.textContent.trim();
}

function prepareLabels() {
    library.querySelectorAll(".token").forEach(token => {
        token.tabIndex = 0;
        token.setAttribute("role", "group");
        token.setAttribute("aria-label", label(token) + ". Double-click or press Enter to replace.");
        token.title = "Double-click to replace or swap this reference.";
    });
}

function chooseReplacement(token) {
    const input = library.querySelector("input");
    input?.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape", code: "Escape", bubbles: true}));
    input?.blur();
    trigger("click", { action: "replace", name: label(token) });
}

library.addEventListener("dblclick", event => {
    const token = event.target.closest(".token");
    if (!token || event.target.closest(".token-remove")) return;
    event.preventDefault();
    event.stopPropagation();
    chooseReplacement(token);
}, true);

library.addEventListener("keydown", event => {
    const token = event.target.closest(".token");
    if (!token || event.target.closest(".token-remove")) return;
    const direction = {ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1}[event.key];
    if (event.altKey && direction) {
        event.preventDefault();
        event.stopPropagation();
        trigger("click", { name: label(token), direction });
    } else if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        chooseReplacement(token);
    }
});

new MutationObserver(prepareLabels).observe(library, { childList: true, subtree: true, characterData: true });
prepareLabels();
