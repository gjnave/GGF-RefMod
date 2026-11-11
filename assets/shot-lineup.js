let playback = {playing:false,text:""};
function showPlayback() {
    const status=element.querySelector('.lineup-play-status');
    if (status) status.textContent=playback.text;
    const stop=element.querySelector('[data-stop-lineup]');
    if (stop) stop.disabled=!playback.playing;
}
document.addEventListener('refmod-lineup-state',event=>{playback=event.detail;showPlayback();});
element.addEventListener('click',event=>{
    const button=event.target.closest('button');
    if (!button || button.disabled) return;
    if (button.hasAttribute('data-play-lineup')) {
        document.dispatchEvent(new CustomEvent('refmod-lineup-play',{detail:[...element.querySelectorAll('li[data-key]')].map(row=>row.dataset.key)}));
    } else if (button.hasAttribute('data-stop-lineup')) {
        document.dispatchEvent(new CustomEvent('refmod-lineup-stop'));
    } else if (button.dataset.action) {
        document.dispatchEvent(new CustomEvent('refmod-lineup-stop'));
        trigger('click',{action:button.dataset.action,key:button.closest('li')?.dataset.key,direction:Number(button.dataset.direction)});
    }
});
watch('value',showPlayback);
showPlayback();
