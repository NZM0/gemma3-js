export const DEFAULT_MODEL_URL = 'https://huggingface.co/NZM0/gemma3-js-1b-it/resolve/main/';

export function createLocalModelSource(files) {
    const entries = new Map(Array.from(files, file => [
        (file.webkitRelativePath || file.name).split('/').slice(1).join('/') || file.name, file
    ]));
    return name => {
        if (!entries.has(name)) { throw new Error(`Missing file: ${name}`); }
        return entries.get(name);
    };
}

export function createRemoteModelSource(base = DEFAULT_MODEL_URL, fetcher = fetch) {
    const root = new URL(base.endsWith('/') ? base : `${base}/`);
    if (root.protocol !== 'https:') { throw new Error('Remote model URL must use HTTPS'); }
    return name => {
        const parts = name.split('/');
        if (parts.some(part => !part || part === '.' || part === '..')) {
            throw new Error(`Invalid model path: ${name}`);
        }
        const url = new URL(parts.map(encodeURIComponent).join('/'), root);
        async function request(range) {
            const response = await fetcher(url.href, range ? {
                headers: { Range: `bytes=${range[0]}-${range[1] - 1}` }
            } : {});
            if (!response.ok) { throw new Error(`${name}: HTTP ${response.status}`); }
            if (range) {
                const match = /^bytes (\d+)-(\d+)\/(\d+|\*)$/.exec(response.headers.get('Content-Range') || '');
                if (response.status !== 206 || !match || Number(match[1]) !== range[0] || Number(match[2]) !== range[1] - 1) {
                    await response.body?.cancel();
                    throw new Error(`${name}: server must support exact byte ranges (HTTP 206)`);
                }
            }
            return response;
        }
        return {
            async text() { return (await request()).text(); },
            async arrayBuffer() { return (await request()).arrayBuffer(); },
            slice(start, end) {
                return { async arrayBuffer() {
                    const data = await (await request([start, end])).arrayBuffer();
                    if (data.byteLength !== end - start) { throw new Error(`${name}: truncated byte range`); }
                    return data;
                } };
            }
        };
    };
}
