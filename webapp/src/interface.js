
async function speakToLlama() {
    const userInput = document.getElementById("speechInput").value;
    const responseElement = document.getElementById("llamaResponse");

    // Clear previous response and input
    responseElement.innerText = "";
    document.getElementById("speechInput").value = "";

    try {
        const response = await fetch('https://ollama.juliancoy.us/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: "llama3.2",
                prompt: userInput,
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        while (true) {
            const { value, done } = await reader.read();

            if (done) break;

            // Decode the chunk and split by newlines to handle multiple JSON objects
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
                try {
                    const jsonResponse = JSON.parse(line);
                    if (jsonResponse.response) {
                        fullResponse += jsonResponse.response;
                        responseElement.innerText = fullResponse;
                    }
                } catch (parseError) {
                    console.warn("Failed to parse JSON chunk:", parseError);
                }
            }
        }

    } catch (error) {
        console.error("Error:", error);
        responseElement.innerText = `Error: ${error.message}`;
    }
} 


let currentIntersectedModel = null;

AFRAME.registerComponent('improved-raycaster-handler', {
    init: function() {
        this.speechInput = document.getElementById('speechInputContainer');
        this.debugInfo = document.getElementById('debugInfo');
        this.isLookingAtCow = false;
        this.angleThreshold = 0.5;
        
        if (this.speechInput) {
            this.speechInput.style.display = 'none';
        }
        
        this.el.addEventListener('raycaster-intersection', (evt) => {
            const intersection = evt.detail.intersections[0];
            this.debugLog('Intersection detected');
            
            if (intersection && intersection.object) {
                const el = intersection.object.el;
                const model = el.getAttribute('gltf-model');
                currentIntersectedModel = model;
                this.debugLog(`Intersected model: ${model}`);
                
                if (model === './assets/freudcow.gltf') {
                    this.isLookingAtCow = true;
                    this.forceShowWindow();
                    this.debugLog('Looking at cow: true');
                }
            }
        });

        this.el.addEventListener('raycaster-intersection-cleared', () => {
            this.isLookingAtCow = false;
            currentIntersectedModel = null;
            this.forceHideWindow();
            this.debugLog('Intersection cleared');
        });

        this.forceHideWindow();
        this.debugLog('Raycaster handler initialized');
    },

    forceShowWindow: function() {
        if (this.speechInput) {
            this.speechInput.style.display = 'block';
            this.speechInput.style.visibility = 'visible';
            this.speechInput.style.opacity = '1';
            this.debugLog('Forcing window to show');
        }
    },

    forceHideWindow: function() {
        if (this.speechInput) {
            this.speechInput.style.display = 'none';
            this.speechInput.style.visibility = 'hidden';
            this.speechInput.style.opacity = '0';
            this.debugLog('Forcing window to hide');
        }
    },

    debugLog: function(message) {
        if (this.debugInfo) {
            this.debugInfo.textContent = message;
            console.log(message);
        }
    }
});

// Add event listener for Enter key
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('speechInput');
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            speakToLlama();
        }
    });
});