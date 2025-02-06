// Function to predict using Linear Regression model
function predictLR() {
    const date = document.getElementById('dateInput').value;

    if (!date) {
        alert("Please enter a valid date.");
        return;
    }

    fetch('/predict_lr', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ date })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('lrResult').innerText = `Linear Regression Prediction: ${data.prediction}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to predict using LSTM model
function predictLSTM() {
    const date = document.getElementById('dateInput').value;

    if (!date) {
        alert("Please enter a valid date.");
        return;
    }

    fetch('/predict_lstm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ date })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('lstmResult').innerText = `LSTM Prediction: ${data.prediction}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

