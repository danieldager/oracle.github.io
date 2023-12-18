// Initialize variables
var index = 0
var data = {}
var plog = '' // prediction log
var klog = '' // keypresses log
var correct = 0
var time = 0
var delay = 0
var uuid = "None"
var batch = []


// Get token from database
async function getToken() {
    fetch('/get', { method: 'GET' })
    .then(response => response.json())
    .then(response => { uuid = response.uuid })
}
await getToken()

// Send datapoints to database
async function postBatch(batch) {
    fetch('/post', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json',},
        body: JSON.stringify({ 
            uuid: uuid,
            data: batch,
         })
    })
      .then(response => response.json())
      .then(response => console.log("log", response))
      .catch(error => console.error('Error fetching data:', error));
}

// Orignal Aaronson Oracle
document.addEventListener('keydown', function(event) {

    // Ignore all other keypresses
    if (!(event.key == 'ArrowLeft' || event.key == 'ArrowRight')) { return }

    // Initialize timer and track delays between presses
    if (time != 0) { delay = Date.now() - time }
    time = Date.now()

    // Make a prediction before the user makes their move
    let prediction
    let prev4 = klog.slice(index - 4, index)

    if (index > 3 && prev4 in data) {
        if (data[prev4].L > data[prev4].R) { prediction = 'L' }
        else if (data[prev4].L < data[prev4].R) { prediction = 'R' }
        else { prediction = ['L', 'R'][Math.floor(Math.random() * 2)] } // duplicate code
    
    }
    else { prediction = ['L', 'R'][Math.floor(Math.random() * 2)] }     // duplicate code
    plog += prediction

    // Log the user's decision
    let key
    if (event.key == 'ArrowLeft') { key = 'L' }
    if (event.key == 'ArrowRight') { key = 'R' }
    klog += key
    index += 1

    // Update predictions with most recent sequence
    if (index > 3) {
        if (prev4 in data) {
            data[prev4][key] += 1
        }
        else {
            data[prev4] = {L: 0, R: 0}
            data[prev4][key] += 1
        }
    }

    // Update trial in database
    var datum = {
        key: key,
        delay: delay,
    }
    batch.push(datum)

    if (index % 30 == 0) {
        postBatch(batch)
        batch = []
    }

    // Calculate average for correct predictions
    if (key == prediction) { correct += 1 }
    let average = (correct / index) * 100

    // Add spaces between for readability
    document.getElementById('predictions').textContent = plog.split('').join(' ')
    document.getElementById('keypresses').textContent = klog.split('').join(' ')
    document.getElementById('average').textContent = Math.floor(average) + "%"
});
