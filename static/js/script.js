document.addEventListener('DOMContentLoaded', function() {
    // Helper functions for UI updates
    function updateUI(elementId, message, type = 'info') {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const icons = {
            info: '<i class="fas fa-info-circle"></i>',
            success: '<i class="fas fa-check-circle"></i>',
            error: '<i class="fas fa-exclamation-triangle"></i>',
            warning: '<i class="fas fa-spinner fa-spin"></i>'
        };
        
        element.innerHTML = `
            <div class="${type}-message">
                <p style='color: #343a40; font-weight: bold;'>${icons[type]} ${message}</p>
            </div>
        `;
    }

    // Progress bar management
    let progressInterval = null;
    
    function startProgressPolling() {
        const progressBarContainer = document.getElementById("video-progress-bar-container");
        const progressBar = document.getElementById("video-progress-bar");
        progressBarContainer.style.display = "block";
        progressBar.style.width = "0%";
        progressBar.innerText = "0%";
        progressBar.setAttribute("aria-valuenow", 0);

        updateProgress();
        progressInterval = setInterval(updateProgress, 500);
    }

    async function updateProgress() {
        try {
            const response = await fetch("/video_progress");
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            
            const data = await response.json();
            const progressBar = document.getElementById("video-progress-bar");
            
            progressBar.style.width = data.progress + '%';
            progressBar.innerText = data.progress + '%';
            progressBar.setAttribute("aria-valuenow", data.progress);
            
            if (data.eta_formatted) {
                updateUI("video-result", `Processing video: ${data.progress}% complete. Estimated time remaining: ${data.eta_formatted}`, "info");
            } else {
                updateUI("video-result", `Processing video: ${data.progress}% complete`, "info");
            }
        } catch (error) {
            console.error('Error fetching progress:', error);
            updateUI("video-result", `Error updating progress: ${error.message}`, "error");
        }
    }

    function stopProgressPolling() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }

    // Tab Navigation
    const tabButtons = document.querySelectorAll('.tab-button');
    const detectionSections = document.querySelectorAll('.detection-section');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            detectionSections.forEach(section => section.classList.remove('active'));
            button.classList.add('active');
            const targetId = button.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Real-time detection setup
    document.getElementById("start-btn").addEventListener("click", () => {
        const videoFeed = document.getElementById("video-feed");
        if (!videoFeed) {
            console.error('Video feed element not found');
            return;
        }

        updateUI("realtime-result", "Starting webcam detection...", "info");
        
        videoFeed.onerror = function() {
            updateUI("realtime-result", "Error accessing webcam. Please check your camera settings.", "error");
            videoFeed.style.display = 'none';
        };

        videoFeed.onload = function() {
            updateUI("realtime-result", "Detection running...", "info");
        };

        let webcamTimeout = setTimeout(() => {
            if (videoFeed.src.includes("/start_detection") && videoFeed.readyState === 0) {
                updateUI("realtime-result", "Webcam connection timed out. Please check your camera connection and try again.", "error");
                videoFeed.style.display = 'none';
            }
        }, 10000);

        videoFeed.addEventListener('loadeddata', function() {
            clearTimeout(webcamTimeout);
        }, { once: true });

        videoFeed.src = "/start_detection";
        videoFeed.style.display = 'block';
    });

    // Stop button
    document.getElementById("stop-btn").addEventListener("click", () => {
        const videoFeed = document.getElementById("video-feed");
        videoFeed.src = "about:blank";
        videoFeed.style.display = 'none';
        updateUI("realtime-result", "Stopping detection...", "info");

        fetch("/stop_detection")
            .then(response => response.ok ? response.json() : Promise.reject(response))
            .then(data => {
                console.log(data.message);
                updateUI("realtime-result", data.message, "success");
            })
            .catch(error => {
                console.error('Error stopping detection:', error);
                updateUI("realtime-result", "Error stopping detection. Please try again or refresh the page.", "error");
            });
    });

    // Clear button (Real-time)
    document.getElementById("clear-realtime-btn").addEventListener("click", () => {
        const videoFeed = document.getElementById("video-feed");
        const originalOnError = videoFeed.onerror;
        videoFeed.onerror = null;
        
        videoFeed.src = "about:blank";
        videoFeed.style.display = 'none';
        document.getElementById("realtime-result").innerHTML = "";
        
        setTimeout(() => {
            videoFeed.onerror = originalOnError;
        }, 100);
    });

    // Image detection
    document.getElementById("image-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        updateUI("image-result", "Processing image... Please wait.", "info");

        try {
            const response = await fetch("/detect-image", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            if (result.output_filename) {
                updateUI("image-result", "Object Detected Successfully.", "success");
                const detectedImage = document.getElementById("detected-image");
                detectedImage.src = `/outputs/${result.output_filename}`;
                detectedImage.style.display = "block";
                
                const downloadLink = document.getElementById("download-image");
                downloadLink.href = `/outputs/${result.output_filename}`;
                downloadLink.style.display = "block";
            } else {
                updateUI("image-result", `Error: ${result.error}`, "error");
            }
        } catch (error) {
            updateUI("image-result", `Error processing image: ${error.message}`, "error");
        }
    });

    // Clear image button
    document.getElementById("clear-image-btn").addEventListener("click", () => {
        document.getElementById("image-input").value = "";
        document.getElementById("image-result").innerHTML = "";
        document.getElementById("detected-image").style.display = "none";
        document.getElementById("download-image").style.display = "none";
    });

    // Video detection
    document.getElementById("video-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        stopProgressPolling();
        updateUI("video-result", "Video processing started...", "info");
        
        document.getElementById("video-progress-bar-container").style.display = "block";
        document.getElementById("detected-video").style.display = "none";
        document.getElementById("download-video").style.display = "none";

        const formData = new FormData(e.target);

        try {
            startProgressPolling();
            const response = await fetch("/detect-video", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            stopProgressPolling();

            if (result.output_filename) {
                updateUI("video-result", "Video processing completed successfully!", "success");
                
                const detectedVideo = document.getElementById("detected-video");
                detectedVideo.src = `/outputs/${result.output_filename}?t=${new Date().getTime()}`;
                detectedVideo.style.display = "block";
                
                detectedVideo.addEventListener('error', function(e) {
                    console.error('Video loading error:', e);
                    updateUI("video-result", "Error loading video. Please try again.", "error");
                });

                detectedVideo.addEventListener('loadeddata', function() {
                    console.log('Video loaded successfully');
                });

                const downloadVideo = document.getElementById("download-video");
                downloadVideo.href = `/outputs/${result.output_filename}`;
                downloadVideo.style.display = "block";

                const progressBar = document.getElementById("video-progress-bar");
                progressBar.style.width = '100%';
                progressBar.innerText = '100%';
                progressBar.setAttribute("aria-valuenow", 100);
            } else {
                updateUI("video-result", `Error: ${result.error}`, "error");
                document.getElementById("video-progress-bar-container").style.display = "none";
            }
        } catch (error) {
            stopProgressPolling();
            updateUI("video-result", `Error: ${error.message}`, "error");
            document.getElementById("video-progress-bar-container").style.display = "none";
        }
    });

    // Stop video button
    document.getElementById("stop-video-btn").addEventListener("click", () => {
        updateUI("video-result", "Stop requested. Finishing current frame...", "warning");
        
        fetch("/stop_detection")
            .then(response => response.ok ? response.json() : Promise.reject(response))
            .then(data => {
                console.log(data.message);
                updateUI("video-result", data.message, "success");
            })
            .catch(error => {
                console.error('Error stopping detection:', error);
                updateUI("video-result", `Error stopping detection: ${error.message}`, "error");
            });
    });

    document.getElementById("clear-video-btn").addEventListener("click", () => {
        document.getElementById("video-input").value = "";
        document.getElementById("video-result").innerHTML = "";
        document.getElementById("detected-video").style.display = "none";
        document.getElementById("download-video").style.display = "none";
        // Hide progress bar on clear
        document.getElementById("video-progress-bar-container").style.display = "none";
        document.getElementById("video-progress-bar").style.width = "0%";
        document.getElementById("video-progress-bar").innerText = "0%";
    });
}); 