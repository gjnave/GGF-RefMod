const library = document.querySelector("#refmod-library");
let pendingOrder = null;

function label(token) {
    return token?.querySelector("span")?.textContent.trim();
}

function prepareLabels() {
    const tokens = [...library.querySelectorAll(".token")];
    if (pendingOrder && JSON.stringify(tokens.map(label)) === JSON.stringify(pendingOrder)) pendingOrder = null;
    tokens.forEach((token, index) => {
        token.tabIndex = 0;
        token.setAttribute("role", "group");
        token.setAttribute("aria-label", label(token) + ". Reference " + (index + 1) + ". Use arrows to reorder or double-click to replace.");
        token.title = "Use the arrows to move this reference. Double-click its name to replace or swap.";
        let controls = token.querySelector(".refmod-moves");
        if (!controls) {
            controls = document.createElement("div");
            controls.className = "refmod-moves";
            for (const [direction, text] of [[-1, "↑"], [1, "↓"]]) {
                const button = document.createElement("button");
                button.type = "button";
                button.dataset.direction = direction;
                button.textContent = text;
                controls.append(button);
            }
            token.insertBefore(controls, token.querySelector(".token-remove"));
        }
        for (const button of controls.querySelectorAll("button")) {
            const direction = Number(button.dataset.direction);
            const title = "Move " + label(token) + (direction === -1 ? " up" : " down");
            button.title = title;
            button.setAttribute("aria-label", title);
            button.disabled = !!pendingOrder || (direction === -1 ? index === 0 : index === tokens.length - 1);
        }
    });
}

function move(token, direction) {
    if (pendingOrder) return;
    const names = [...library.querySelectorAll(".token")].map(label);
    const index = names.indexOf(label(token));
    const target = index + direction;
    if (index < 0 || target < 0 || target >= names.length) return;
    [names[index], names[target]] = [names[target], names[index]];
    pendingOrder = names;
    prepareLabels();
    trigger("click", { name: label(token), direction });
    // Unlock on a failed request too; successful updates unlock when the labels change.
    setTimeout(() => { if (pendingOrder === names) { pendingOrder = null; prepareLabels(); } }, 10000);
}

// Stop the dropdown from treating an arrow click as a selection or removal.
for (const type of ["pointerdown", "mousedown", "click", "dblclick"]) {
    library.addEventListener(type, event => {
        const button = event.target.closest(".refmod-moves button");
        if (!button) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        if (type === "click" && !button.disabled) move(button.closest(".token"), Number(button.dataset.direction));
    }, true);
}

function chooseReplacement(token) {
    const input = library.querySelector("input");
    input?.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape", code: "Escape", bubbles: true}));
    input?.blur();
    trigger("click", { action: "replace", name: label(token) });
}

library.addEventListener("dblclick", event => {
    const token = event.target.closest(".token");
    if (!token || event.target.closest(".token-remove, .refmod-moves")) return;
    event.preventDefault();
    event.stopPropagation();
    chooseReplacement(token);
}, true);

library.addEventListener("keydown", event => {
    const token = event.target.closest(".token");
    if (!token || event.target.closest(".token-remove, .refmod-moves")) return;
    const direction = {ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1}[event.key];
    if (event.altKey && direction) {
        event.preventDefault();
        event.stopPropagation();
        move(token, direction);
    } else if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        chooseReplacement(token);
    }
});

new MutationObserver(prepareLabels).observe(library, { childList: true, subtree: true, characterData: true });
prepareLabels();
