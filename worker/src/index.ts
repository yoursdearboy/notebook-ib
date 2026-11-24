export interface Env {
  UPSTREAM_URL: string;
  UPSTREAM_TOKEN_HEADER: string;
  UPSTREAM_TOKEN: string;
  CLIENT_TOKEN?: string;
}

const HOP_BY_HOP = [
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
];

function timingSafeEqual(a: string, b: string): boolean {
  const enc = new TextEncoder();
  const x = enc.encode(a);
  const y = enc.encode(b);
  if (x.byteLength !== y.byteLength) return false;
  return crypto.subtle.timingSafeEqual(x, y);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const incoming = new URL(request.url);
    const isLocal = ["localhost", "127.0.0.1", "[::1]"].includes(incoming.hostname);
    if (incoming.protocol === "http:" && !isLocal) {
      incoming.protocol = "https:";
      return Response.redirect(incoming.toString(), 308);
    }

    if (!env.UPSTREAM_TOKEN) {
      return new Response("UPSTREAM_TOKEN is not configured", { status: 500 });
    }

    const headers = new Headers(request.headers);

    if (env.CLIENT_TOKEN) {
      const auth = headers.get("Authorization") ?? "";
      const presented = auth.startsWith("Bearer ") ? auth.slice(7) : "";
      if (!timingSafeEqual(presented, env.CLIENT_TOKEN)) {
        return new Response("Unauthorized", {
          status: 401,
          headers: { "WWW-Authenticate": "Bearer" },
        });
      }
    }

    const upstream = new URL(env.UPSTREAM_URL);
    const basePath = upstream.pathname.replace(/\/$/, "");
    upstream.pathname = basePath + incoming.pathname;
    upstream.search = incoming.search;

    const isWebSocket = headers.get("Upgrade")?.toLowerCase() === "websocket";
    for (const name of HOP_BY_HOP) headers.delete(name);
    if (isWebSocket) headers.set("Upgrade", "websocket");
    // MicroVM ingress drops Cookie. Nginx restores it from this header.
    headers.delete("X-Notebook-WebUI-Cookie");
    const cookie = headers.get("Cookie");
    if (cookie) headers.set("X-Notebook-WebUI-Cookie", cookie);
    // Never leak the client's credentials upstream; replace with the microVM token.
    headers.delete("Authorization");
    const tokenHeader = env.UPSTREAM_TOKEN_HEADER || "Authorization";
    headers.set(
      tokenHeader,
      tokenHeader.toLowerCase() === "authorization"
        ? `Bearer ${env.UPSTREAM_TOKEN}`
        : env.UPSTREAM_TOKEN,
    );
    headers.set("X-Forwarded-Host", incoming.host);
    headers.set("X-Forwarded-Proto", incoming.protocol.replace(":", ""));
    const clientIp = request.headers.get("CF-Connecting-IP");
    if (clientIp) headers.set("X-Forwarded-For", clientIp);

    const response = await fetch(upstream.toString(), {
      method: request.method,
      headers,
      body: request.body,
      redirect: "manual",
    });

    if (response.webSocket) return response;

    // Pass the body through as a stream so SSE (stream=true) responses are not buffered.
    const out = new Headers(response.headers);
    for (const name of HOP_BY_HOP) out.delete(name);
    out.delete("Set-Cookie");
    for (const cookie of response.headers.getAll("Set-Cookie")) {
      if (!cookie.startsWith("__cf_bm=")) out.append("Set-Cookie", cookie);
    }
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: out,
    });
  },
} satisfies ExportedHandler<Env>;
