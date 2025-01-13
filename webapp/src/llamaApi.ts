export const sendMessageToLlamaAPI = async (
    model: string,
    context: string,
    prompt: string
): Promise<ReadableStream<Uint8Array>> => {
    const jsonData = {
        model,
        prompt: `${context}\nYou: ${prompt}`,
    };

    const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(jsonData),
    });

    if (!response.body) {
        throw new Error('No response body from Llama API');
    }

    return response.body;
};

export const streamLlamaResponse = async (
    reader: ReadableStreamDefaultReader<Uint8Array>,
    onData: (data: string) => void,
    onError: (error: Error) => void
) => {
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let endOfLineIndex;

        while ((endOfLineIndex = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, endOfLineIndex);
            buffer = buffer.slice(endOfLineIndex + 1);

            if (line.trim()) {
                try {
                    const parsedLine = JSON.parse(line);
                    onData(parsedLine.response);
                } catch (e) {
                    console.error('Error parsing Llama response line:', e);
                }
            }
        }
    }
};
