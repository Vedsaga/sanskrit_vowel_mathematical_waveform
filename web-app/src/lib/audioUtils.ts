
export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    return await audioContext.decodeAudioData(arrayBuffer);
}

export function getWindingPoints(
    audioBuffer: AudioBuffer,
    frequency: number,
    numPoints: number = 1000,
    startTime: number = 0,
    duration: number = 0.1
): { x: number; y: number }[] {
    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    const startSample = Math.floor(startTime * sampleRate);
    const endSample = Math.min(startSample + Math.floor(duration * sampleRate), channelData.length);
    
    const points: { x: number; y: number }[] = [];
    const step = Math.max(1, Math.floor((endSample - startSample) / numPoints));
    
    for (let i = startSample; i < endSample; i += step) {
        const t = i / sampleRate;
        const amplitude = channelData[i];
        const angle = -2 * Math.PI * frequency * t;
        
        points.push({
            x: amplitude * Math.cos(angle),
            y: amplitude * Math.sin(angle)
        });
    }
    
    return points;
}
