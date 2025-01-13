const wsUrl = 'ws://compai.local:8188/'; // Replace with your WebSocket server URL
const pc = new RTCPeerConnection();
let audioElement;
let sessionId;
let ws;
const messageQueue = [];

// Function to connect to the WebSocket server and initialize session
async function connectToRoom() {
    console.log('Attempting to connect to WebSocket at:', wsUrl);
    ws = new WebSocket(wsUrl, 'janus-protocol');

    ws.onopen = () => {
        console.log('Connected to WebSocket');
        
        // Create a new session
        const createSessionRequest = {
            janus: 'create',
            transaction: 'create_' + Math.random().toString(36).substring(7)
        };
        sendMessage(JSON.stringify(createSessionRequest));
    };

    ws.onerror = error => console.error('WebSocket error:', error);
    ws.onclose = () => console.log('WebSocket closed');

    ws.onmessage = async event => {
        console.log('Received message from WebSocket:', event.data);
        const data = JSON.parse(event.data);

        // Handle session creation response
        if (data.janus === 'success' && data.transaction.startsWith('create_')) {
            console.log('Session created. ID:', data.data.id);
            sessionId = data.data.id;

            // Now, join the room
            const joinRoomRequest = {
                janus: 'attach',
                session_id: sessionId,
                transaction: 'join_' + Math.random().toString(36).substring(7),
                plugin: 'janus.plugin.audiobridge' // or the appropriate plugin for your setup
            };
            sendMessage(JSON.stringify(joinRoomRequest));
        }

        // Handle room join response
        if (data.janus === 'success' && data.transaction.startsWith('join_')) {
            console.log('Joined the room successfully. Waiting for SDP offer...');
        }

        // Handle offer from the Audiobridge plugin
        if (data.janus === 'event' && data.plugindata && data.plugindata.data && data.plugindata.data.audiobridge === 'event') {
            // Check if there's an SDP offer in the response
            if (data.jsep) {
                console.log('Received SDP offer from server, setting remote description...');
                const sdp = data.jsep;
                await pc.setRemoteDescription(new RTCSessionDescription(sdp));

                console.log('Creating answer...');
                const answer = await pc.createAnswer();
                await pc.setLocalDescription(answer);

                console.log('Sending answer back to server...');
                const response = {
                    janus: 'message',
                    session_id: sessionId,
                    transaction: data.transaction,
                    body: { request: 'answer' },
                    jsep: answer
                };
                sendMessage(JSON.stringify(response));
            } else {
                console.log('No SDP offer received yet. Waiting...');
            }
        }
    };

    // Set up the peer connection to handle incoming audio track
    pc.ontrack = event => {
        console.log('Received audio track from peer connection');
        if (!audioElement) {
            console.log('Initializing audio element to play received stream...');
            audioElement = document.createElement('audio');
            audioElement.controls = true;
            audioElement.autoplay = true;
            audioElement.srcObject = event.streams[0];
            document.body.appendChild(audioElement);
            console.log('Audio stream started');
        }
    };
}

// Function to send a message, ensuring WebSocket is in OPEN state
function sendMessage(data) {
    if (ws.readyState === WebSocket.OPEN) {
        console.log('Sending message:', data);
        ws.send(data);
    } else {
        console.log('WebSocket not open. Queuing message:', data);
        messageQueue.push(data); // Queue the message if WebSocket isn't open
    }
}

// Call the function to connect to the room
connectToRoom().catch(error => console.error('Connection failed:', error));
