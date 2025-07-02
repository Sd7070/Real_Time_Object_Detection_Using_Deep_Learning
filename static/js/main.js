// Webcam Control Functions
let videoStream = null;

// Initialize the tabs
$(document).ready(function() {
    $('.tab-button').click(function() {
        const target = $(this).data('target');
        $('.tab-button').removeClass('active');
        $(this).addClass('active');
        $('.detection-section').removeClass('active');
        $(`#${target}`).addClass('active');
    });

    // Initialize the video feed
    initializeVideoFeed();
});

function initializeVideoFeed() {
    const videoFeed = document.getElementById('video-feed');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const clearBtn = document.getElementById('clear-realtime-btn');

    startBtn.addEventListener('click', async () => {
        try {
            // Start the video feed
            videoFeed.src = '/video_feed';
            videoFeed.play();
            
            // Start object detection
            fetch('/start_detection', { method: 'POST' });
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            clearBtn.disabled = false;
        } catch (error) {
            console.error('Error starting video feed:', error);
            alert('Failed to start video feed. Please check your webcam connection.');
        }
    });

    stopBtn.addEventListener('click', () => {
        // Stop the video feed
        videoFeed.src = '';
        
        // Stop object detection
        fetch('/stop_detection', { method: 'POST' });
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        clearBtn.disabled = true;
    });

    clearBtn.addEventListener('click', () => {
        // Clear the detection results
        const resultDiv = document.getElementById('realtime-result');
        resultDiv.innerHTML = '';
    });
}
