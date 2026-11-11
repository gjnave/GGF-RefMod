element.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-ref]");
    const prompt = document.querySelector("#scene-and-sound textarea");
    if (!button || !prompt) return;

    const start = prompt.selectionStart;
    const end = prompt.selectionEnd;
    const left = prompt.value.slice(0, start);
    const right = prompt.value.slice(end);
    const prefix = left && !/[\s([{]$/.test(left) ? " " : "";
    const suffix = !right || /^[\p{L}\p{N}<]/u.test(right) ? " " : "";
    const phrase = button.dataset.ref.startsWith("Audio ")
        ? "Using the exact voice, tone, and vocal characteristics from <" + button.dataset.ref + ">"
        : "The Character from <" + button.dataset.ref + ">";
    prompt.setRangeText(prefix + phrase + suffix, start, end, "end");
    prompt.dispatchEvent(new Event("input", { bubbles: true }));
    prompt.focus();
});
