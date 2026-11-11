const pinKey = () => "minimax-refmod-gallery:" + localStorage.getItem("minimax-refmod-workspace");
const limitKey = "minimax-refmod-gallery-limit";
const pageKey = () => pinKey() + ":page";
const loopKey = "minimax-refmod-player-loop";
const continuousKey = "minimax-refmod-player-continuous";
const filterKey = () => pinKey() + ":filter";
const favoritePending = new Map();
const lineupPending = new Map();
let lineupKeys = [], lineupIndex = -1;
let inspectingUntil=0;

function lineupStatus(text) {
    document.dispatchEvent(new CustomEvent('refmod-lineup-state',{detail:{playing:lineupIndex>=0,text}}));
}

function stopLineup(pause=false,text='') {
    lineupKeys=[];lineupIndex=-1;
    if (pause) document.querySelector('#clip-player video')?.pause();
    lineupStatus(text);
}

function playLineupAt(index) {
    const tile=[...element.querySelectorAll('.clip-tile')].find(item=>item.dataset.key===lineupKeys[index]);
    if (!tile) { stopLineup(true,'A lineup shot is no longer available.');return; }
    lineupIndex=index;
    display(tile);
    const player=document.querySelector('#clip-player video');
    player.currentTime=0;
    lineupStatus('Playing shot '+(index+1)+' of '+lineupKeys.length);
    player.play().catch(error=>{
        if (error.name!=='AbortError' && lineupIndex===index) lineupStatus('Press play to continue the lineup.');
    });
}

document.addEventListener('refmod-lineup-stop',()=>stopLineup(true));
document.addEventListener('refmod-lineup-play',event=>{
    if (!Array.isArray(event.detail) || !event.detail.length) return;
    lineupKeys=event.detail;lineupIndex=0;
    localStorage.setItem(continuousKey,'false');localStorage.setItem(loopKey,'false');
    playLineupAt(0);
});

function requestLineup(tile,include) {
    if (!tile || lineupPending.has(tile.dataset.key)) return;
    const key=tile.dataset.key;
    stopLineup(false);
    lineupPending.set(key,include);
    display(null);
    trigger('click',{action:include?'lineup-add':'lineup-remove',key});
    setTimeout(()=>{lineupPending.delete(key);display(null);},8000);
}

function matchesFilter(tile) {
    return localStorage.getItem(filterKey()) !== "favorites" || tile.dataset.favorite === "true";
}

function requestFavorite(tile) {
    if (!tile || favoritePending.has(tile.dataset.key)) return;
    const key = tile.dataset.key;
    const favorite = tile.dataset.favorite !== "true";
    favoritePending.set(key,favorite);
    display(null);
    trigger("click", {action:"favorite",key,favorite});
    setTimeout(() => { favoritePending.delete(key); display(null); },8000);
}

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
    trigger("click", {action:"delete",key: tile.dataset.key, destination, pinned: sessionStorage.getItem(pinKey()) || ""});
    // Restore playback controls if another process prevented the move.
    setTimeout(() => display(null), 1800);
}

function display(clicked) {
    const allTiles = Array.from(element.querySelectorAll(".clip-tile"));
    const tiles = allTiles.filter(matchesFilter);
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
    const select = panel.querySelector('select[aria-label="Thumbnails shown"]');
    const filter = panel.querySelector('select[aria-label="Filter outputs"]');
    filter.value = localStorage.getItem(filterKey()) === "favorites" ? "favorites" : "all";
    filter.onchange = () => {
        localStorage.setItem(filterKey(),filter.value);
        sessionStorage.setItem(pageKey(),"0");
        display(null);
    };
    const remove = panel.querySelector(".delete-current");
    const favoriteCurrent = panel.querySelector(".favorite-current");
    const lineupCurrent = panel.querySelector('.lineup-current');
    for (const tile of allTiles) {
        const key = tile.dataset.key;
        if (favoritePending.get(key) === (tile.dataset.favorite === "true")) favoritePending.delete(key);
        tile.parentElement.querySelector(".clip-favorite").disabled = favoritePending.has(key);
        const included=Number(tile.dataset.lineup)>0;
        if (lineupPending.get(key)===included) lineupPending.delete(key);
        const checkbox=tile.parentElement.querySelector('.clip-select input');
        checkbox.checked=lineupPending.get(key)??included;checkbox.disabled=lineupPending.has(key);
    }
    const empty = element.querySelector(".clip-empty");
    if (empty) empty.hidden = filter.value !== "favorites" || tiles.length > 0;
    const repeat = panel.querySelector(".loop-playback");
    const continuous = panel.querySelector(".continuous-playback");
    continuous.checked = localStorage.getItem(continuousKey) === "true";
    player.loop = repeat.checked = !continuous.checked && localStorage.getItem(loopKey) === "true";
    repeat.onchange = () => {
        stopLineup(false);
        player.loop = repeat.checked;
        localStorage.setItem(loopKey,String(repeat.checked));
        if (repeat.checked) {
            continuous.checked = false;
            localStorage.setItem(continuousKey,"false");
        }
    };
    continuous.onchange = () => {
        stopLineup(false);
        localStorage.setItem(continuousKey,String(continuous.checked));
        if (continuous.checked) {
            player.loop = repeat.checked = false;
            localStorage.setItem(loopKey,"false");
        }
    };
    player.onended = () => {
        if (lineupIndex>=0) {
            if (lineupIndex+1>=lineupKeys.length) stopLineup(true,'Lineup finished.');
            else playLineupAt(lineupIndex+1);
            return;
        }
        if (!continuous.checked) return;
        const playlist = Array.from(element.querySelectorAll(".clip-tile")).filter(matchesFilter);
        if (!playlist.length) return;
        const index = playlist.findIndex(tile => tile.dataset.key === player.dataset.key);
        const next = playlist[(index + 1) % playlist.length];
        display(next);
        player.currentTime = 0;
        player.play().catch(error => {
            if (error.name !== "AbortError" && player.dataset.key === next.dataset.key) {
                panel.querySelector(".clip-caption").textContent += " · Press play to continue.";
            }
        });
    };
    const savedLimit = localStorage.getItem(limitKey);
    const showAll = savedLimit === "all";
    const limit = showAll ? Math.max(1,tiles.length) : Math.max(1, Math.min(20, parseInt(savedLimit,10) || 20));
    let page = Math.max(0, parseInt(sessionStorage.getItem(pageKey()),10) || 0);
    select.value = showAll ? "all" : String(limit);
    select.onchange = () => {
        localStorage.setItem(limitKey,select.value);
        sessionStorage.setItem(pageKey(),"0");
        display(null);
    };
    const navigation = document.querySelector("#clip-navigation");
    const pageCount = Math.max(1,Math.ceil(tiles.length / limit));
    page = Math.min(page,pageCount - 1);
    if (navigation) navigation.hidden = tiles.length === 0;
    remove.hidden = favoriteCurrent.hidden = lineupCurrent.hidden = allTiles.length === 0;
    if (!allTiles.length) {
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
    const newest = allTiles[0].dataset.key;
    if (lineupIndex<0 && Array.isArray(seen) && seen.length && !seen.includes(newest)) {
        pinned = newest;
        page = 0;
        sessionStorage.setItem(pinKey(), pinned);
    }
    sessionStorage.setItem(seenKey,JSON.stringify(allTiles.map(tile => tile.dataset.key)));
    if (clicked) {
        pinned = clicked.dataset.key;
        sessionStorage.setItem(pinKey(), pinned);
        const position=tiles.indexOf(clicked);
        if (position>=0) page = Math.floor(position / limit);
    }
    const current = allTiles.find(tile => tile.dataset.key === pinned) || allTiles[0];
    sessionStorage.setItem(pageKey(),String(page));
    const start = page * limit;
    const visible = new Set(tiles.slice(start,start + limit));
    allTiles.forEach(tile => {
        tile.parentElement.hidden = !visible.has(tile);
        tile.setAttribute("aria-pressed",String(tile === current));
    });
    if (navigation) {
        const previous = navigation.querySelector(".clips-prev");
        const next = navigation.querySelector(".clips-next");
        previous.disabled = showAll || page === 0;
        next.disabled = showAll || page >= pageCount - 1;
        previous.onclick = () => {
            sessionStorage.setItem(pageKey(),String(Math.max(0,page - 1)));
            display(null);
        };
        next.onclick = () => {
            sessionStorage.setItem(pageKey(),String(Math.min(pageCount - 1,page + 1)));
            display(null);
        };
        navigation.querySelector(".clip-page-status").textContent = showAll
            ? "All " + tiles.length + " clips"
            : "Page " + (page + 1) + " of " + pageCount + " · " + (start + 1) + "–" + Math.min(start + limit,tiles.length) + " of " + tiles.length;
    }
    player.dataset.key = current.dataset.key;
    remove.onclick = () => requestDelete(current,player);
    favoriteCurrent.textContent = current.dataset.favorite === "true" ? "★ Favorited" : "☆ Favorite";
    favoriteCurrent.setAttribute("aria-pressed",current.dataset.favorite);
    favoriteCurrent.setAttribute("aria-label",current.dataset.favorite === "true" ? "Remove current video from favorites" : "Favorite current video");
    favoriteCurrent.disabled = favoritePending.has(current.dataset.key);
    favoriteCurrent.onclick = () => requestFavorite(current);
    lineupCurrent.textContent=Number(current.dataset.lineup)>0?'Remove from lineup':'Add to lineup';
    lineupCurrent.disabled=lineupPending.has(current.dataset.key);
    lineupCurrent.onclick=()=>requestLineup(current,Number(current.dataset.lineup)===0);
    if (player.getAttribute("src") !== current.dataset.src) {
        player.src = current.dataset.src;
        caption.textContent = current.title;
    }
}

element.addEventListener("click", event => {
    if (event.target.closest('.clip-select')) return;
    if (event.target.closest(".clip-favorite")) {
        requestFavorite(event.target.closest(".clip-item").querySelector(".clip-tile"));
        return;
    }
    if (event.target.closest(".clip-remove")) {
        requestDelete(event.target.closest(".clip-item").querySelector(".clip-tile"),document.querySelector("#clip-player video"));
        return;
    }
    const tile = event.target.closest(".clip-tile");
    if (tile) { stopLineup(false);display(tile); }
});

element.addEventListener('change',event=>{
    const checkbox=event.target.closest('.clip-select input');
    if (checkbox) requestLineup(checkbox.closest('.clip-item').querySelector('.clip-tile'),checkbox.checked);
});
element.addEventListener("dblclick", event => {
    if (event.target.closest(".clip-favorite,.clip-remove,.clip-select")) return;
    const tile=event.target.closest(".clip-tile");
    if (!tile) return;
    display(tile);
    openDetails();
});
watch("value", () => display(null));
display(null);
