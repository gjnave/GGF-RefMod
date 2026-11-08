const pinKey = () => "minimax-refmod-gallery:" + localStorage.getItem("minimax-refmod-workspace");
const limitKey = "minimax-refmod-gallery-limit";
const loopKey = "minimax-refmod-player-loop";
let inspectingUntil=0;

// Chrome's native video controls can enter fullscreen on their own double-click.
// Keep the details panel reachable, without changing the normal fullscreen button.
function leaveVideoFullscreen() {
    if (performance.now()<inspectingUntil && document.fullscreenElement?.matches('#clip-player video')) {
        document.exitFullscreen().catch(()=>{});
    }
}
document.addEventListener('fullscreenchange',leaveVideoFullscreen);

function openDetails() {
    const control=document.querySelector('#inspect-project');
    (control?.matches('button') ? control : control?.querySelector('button'))?.click();
}

function requestDelete(tile, player) {
    if (!tile) return;
    const destination = tile.dataset.destination;
    const label = destination === "bin" ? "the Windows Recycle Bin" : "the Deleted videos folder";
    if (!window.confirm("Move this video to " + label + "?\n\n" + tile.title)) return;
    if (player.dataset.key === tile.dataset.key) {
        player.pause();
        player.removeAttribute("src");
        player.load();
    }
    trigger("click", {key: tile.dataset.key, destination, pinned: sessionStorage.getItem(pinKey()) || ""});
    // Restore playback controls if another process prevented the move.
    setTimeout(() => display(null), 1800);
}

function display(clicked) {
    const tiles = Array.from(element.querySelectorAll(".clip-tile"));
    const panel = document.querySelector("#clip-player");
    if (!panel) return;
    const player = panel.querySelector("video");
    player.ondblclick = event => {
        event.preventDefault();
        if (player.dataset.key) {
            inspectingUntil=performance.now()+1200;
            leaveVideoFullscreen();
            openDetails();
        }
    };
    const caption = panel.querySelector(".clip-caption");
    const select = panel.querySelector("select");
    const remove = panel.querySelector(".delete-current");
    const repeat = panel.querySelector(".loop-playback");
    player.loop = repeat.checked = localStorage.getItem(loopKey) === "true";
    repeat.onchange = () => {
        player.loop = repeat.checked;
        localStorage.setItem(loopKey,String(repeat.checked));
    };
    const limit = Math.max(1, Math.min(20, parseInt(localStorage.getItem(limitKey),10) || 20));
    select.value = String(limit);
    select.onchange = () => {
        localStorage.setItem(limitKey,select.value);
        display(null);
    };
    remove.hidden = tiles.length === 0;
    if (!tiles.length) {
        if (player.hasAttribute("src")) {
            player.removeAttribute("src");
            player.load();
        }
        delete player.dataset.key;
        caption.textContent = "Completed videos appear here.";
        return;
    }
    let pinned = sessionStorage.getItem(pinKey());
    const seenKey = pinKey() + ":seen-clips";
    let seen = [];
    try { seen = JSON.parse(sessionStorage.getItem(seenKey) || "[]"); }
    catch { /* Older or cleared browser state starts with the visible clips. */ }
    const newest = tiles[0].dataset.key;
    if (Array.isArray(seen) && seen.length && !seen.includes(newest)) {
        pinned = newest;
        sessionStorage.setItem(pinKey(), pinned);
    }
    sessionStorage.setItem(seenKey,JSON.stringify(tiles.map(tile => tile.dataset.key)));
    if (clicked) {
        pinned = clicked.dataset.key;
        sessionStorage.setItem(pinKey(), pinned);
    }
    const current = tiles.find(tile => tile.dataset.key === pinned) || tiles[0];
    tiles.forEach((tile,index) => {
        tile.parentElement.hidden = index >= limit;
        tile.setAttribute("aria-pressed",String(tile === current));
    });
    player.dataset.key = current.dataset.key;
    remove.onclick = () => requestDelete(current,player);
    if (player.getAttribute("src") !== current.dataset.src) {
        player.src = current.dataset.src;
        caption.textContent = current.title;
    }
}

element.addEventListener("click", event => {
    if (event.target.closest(".clip-remove")) {
        requestDelete(event.target.closest(".clip-item").querySelector(".clip-tile"),document.querySelector("#clip-player video"));
        return;
    }
    const tile = event.target.closest(".clip-tile");
    if (tile) display(tile);
});
element.addEventListener("dblclick", event => {
    const tile=event.target.closest(".clip-tile");
    if (!tile) return;
    display(tile);
    openDetails();
});
watch("value", () => display(null));
display(null);
