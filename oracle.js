var index = 0
var data = {}
var plog = '' // prediction log
var klog = '' // keypresses log
var correct = 0 

document.addEventListener('keydown', function(event) {
    // Ignore all other keypresses
    if (!(event.key == 'ArrowLeft' || event.key == 'ArrowRight')) return    

    // Make a prediction before the user makes their move
    let prediction
    let prev4 = klog.slice(index - 4, index)

    if (index > 3 && prev4 in data) {
        if (data[prev4].L > data[prev4].R) {
            prediction = 'L'
        }
        else if (data[prev4].L < data[prev4].R) {
            prediction = 'R'
        }
        else {
            prediction = ['L', 'R'][Math.floor(Math.random() * 2)] // duplicate code
        }
    }
    else {
        prediction = ['L', 'R'][Math.floor(Math.random() * 2)]    // duplicate code
    }
    plog += prediction

    // Log the user's decision
    let keypress

    if (event.key == 'ArrowLeft') {
        index += 1
        keypress = 'L'
    }
    else if (event.key == 'ArrowRight') {
        index += 1
        keypress = 'R'
    }
    klog += keypress 

    // Update predictions with most recent sequence
    if (index > 3) {
        if (prev4 in data) {
            data[prev4][keypress] += 1
        }
        else {
            data[prev4] = {L: 0, R: 0}
            data[prev4][keypress] += 1
        }
    }

    // Calculate average for correct predictions
    if (keypress == prediction) {
        correct += 1
    }
    let average = (correct / index) * 100

    // Add spaces between for readability
    document.getElementById('predictions').textContent = plog.split('').join(' ')
    document.getElementById('keypresses').textContent = klog.split('').join(' ')
    document.getElementById('average').textContent = Math.floor(average) + "%"
});
