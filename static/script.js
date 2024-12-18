const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function updateCanvas(x, y) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings
    ctx.fillStyle = 'blue'; // Set color for the point
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2); // Draw the point
    ctx.fill();
}

function fetchCoordinates() {
    fetch('/get_coordinates')
        .then(response => response.json())
        .then(data => {
            updateCanvas(data.x, data.y);
        })
        .catch(error => console.error('Error fetching coordinates:', error));
}

// Update the canvas every second
setInterval(fetchCoordinates, 1000);
