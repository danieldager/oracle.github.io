let keypressLog = '';

document.addEventListener('keydown', function(event) {
    if(event.key == 'ArrowLeft') {
        keypressLog += 'L '
    }
    else if(event.key == 'ArrowRight') {
        keypressLog += 'R '
    }
    document.getElementById('keypresses').textContent = keypressLog;
});
