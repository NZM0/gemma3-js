export const MOBILE_MODEL_URL = 'https://huggingface.co/NZM0/gemma3-js-1b-it-int4/resolve/main/';

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

export function createRemoteModelSource(base = DEFAULT_MODEL_URL, fetcher = fetch, allowLocal = false) {
    const root = new URL(base.endsWith('/') ? base : `${base}/`);
    if (root.protocol !== 'https:' && !(allowLocal && root.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(root.hostname))) { throw new Error('Remote model URL must use HTTPS'); }
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
            async arrayBuffer() {
                try { return await (await request()).arrayBuffer(); }
                catch (error) { throw new Error(`${name}: ${error.message}`, { cause: error }); }
            },
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

export function createBundledModelSource(base, fetcher = fetch) {
    return createRemoteModelSource(base, fetcher, true);
}

// A logical file backed by small, independently fetchable model resources.
export function createChunkedHeadSource(get, chunkBytes, extension = 'f16') {
    return {
        slice(start, end) {
            return { async arrayBuffer() {
                if (start % chunkBytes === 0 && end - start === chunkBytes) {
                    const name = `head-part-${String(start / chunkBytes).padStart(3, '0')}.${extension}`;
                    const data = await get(name).arrayBuffer();
                    if (data.byteLength !== chunkBytes) { throw new Error(`Truncated ${name}`); }
                    return data;
                }
                const result = new Uint8Array(end - start);
                for (let offset = start; offset < end;) {
                    const index = Math.floor(offset / chunkBytes);
                    const within = offset % chunkBytes;
                    const count = Math.min(end - offset, chunkBytes - within);
                    const name = `head-part-${String(index).padStart(3, '0')}.${extension}`;
                    const bytes = new Uint8Array(await get(name).arrayBuffer());
                    if (within + count > bytes.length) { throw new Error(`Truncated ${name}`); }
                    result.set(bytes.subarray(within, within + count), offset - start);
                    offset += count;
                }
                return result.buffer;
            } };
        }
    };
}
