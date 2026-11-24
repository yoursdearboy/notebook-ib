# Open WebUI proxy

The Cloudflare Worker forwards browser requests to the DigitalOcean MicroVM’s HTTPS ingress. The ingress routes to Nginx on `0.0.0.0:8080`, and Nginx proxies to `uv run bin/serve` on `127.0.0.1:8000`.

## MicroVM setup

With the app running on port 8000, install Nginx in the MicroVM as root. On a fresh installation dedicated to this proxy, remove the packaged default site if you only want Nginx to listen on port 8080.

```sh
apt-get update
apt-get install -y nginx
# Optional on a fresh, dedicated installation:
unlink /etc/nginx/sites-enabled/default
```

Save this as `/etc/nginx/conf.d/notebook-ib.conf`.

```nginx
map $http_x_notebook_webui_cookie $notebook_webui_cookie {
    ""      $http_cookie;
    default $http_x_notebook_webui_cookie;
}

map $notebook_webui_cookie $notebook_cookie_authorization {
    default "";
    "~(?:^|;)\s*token=(?<notebook_session_token>[^;\s]+)" "Bearer $notebook_session_token";
}

map $http_authorization $notebook_authorization {
    ""      $notebook_cookie_authorization;
    default $http_authorization;
}

map $http_upgrade $notebook_connection {
    ""      "";
    default upgrade;
}

map $http_x_forwarded_proto $notebook_forwarded_proto {
    default $scheme;
    http    http;
    https   https;
}

map $http_x_forwarded_host $notebook_forwarded_host {
    ""      $http_host;
    default $http_x_forwarded_host;
}

server {
    listen 0.0.0.0:8080;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $notebook_forwarded_host;
        proxy_set_header X-Forwarded-Host $notebook_forwarded_host;
        proxy_set_header X-Forwarded-Proto $notebook_forwarded_proto;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Cookie $notebook_webui_cookie;
        proxy_set_header Authorization $notebook_authorization;
        proxy_set_header X-Notebook-WebUI-Cookie "";
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $notebook_connection;
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```

Run `nginx -t`, then run `nginx` to start it or `nginx -s reload` if it is already running. The proxy supports WebSocket upgrades and streams responses with buffering disabled.

## Worker setup

Confirm that the MicroVM ingress URL routes to port 8080.

```sh
doctl compute microvm url list <microvm-id>
```

Set `UPSTREAM_URL` in `wrangler.jsonc` to that ingress URL. The checked-in URL may point to another MicroVM. Set `UPSTREAM_TOKEN` to an ingress token for the same VM. With the default `UPSTREAM_TOKEN_HEADER` of `Authorization`, the Worker sends that token to DigitalOcean as a bearer token. An optional `CLIENT_TOKEN` requires callers to authenticate to the Worker with a separate bearer token.

From this directory, install dependencies and deploy.

```sh
pnpm install
pnpm exec wrangler secret put UPSTREAM_TOKEN
pnpm deploy
```

## Cookie handling

- DigitalOcean ingress drops the browser’s `Cookie` header. The Worker clears any client-supplied `X-Notebook-WebUI-Cookie` and copies the actual browser cookie into that header. Nginx restores `Cookie` before forwarding the request and removes the transport header. Direct requests to Nginx can use an ordinary cookie.
- Open WebUI can identify the user from a `token` cookie, but its `GET /api/v1/auths/` handler also reads a bearer token. A valid cookie without that header can produce HTTP 500 with `AttributeError: 'NoneType' object has no attribute 'credentials'`. When authorization is absent, Nginx supplies `Authorization: Bearer <session token>` from the restored cookie. An existing authorization header takes precedence.
- On the way back, the Worker forwards the app’s `Set-Cookie` headers and filters out Cloudflare’s `__cf_bm` cookie.

## Verify

Inside the MicroVM, confirm the app and proxy both answer.

```sh
curl -fsS -o /dev/null -w 'App HTTP %{http_code}\n' http://127.0.0.1:8000/
curl -fsS -o /dev/null -w 'Proxy HTTP %{http_code}\n' http://127.0.0.1:8080/
```

For an end-to-end check, sign in through the deployed Worker and reload the page. In the browser’s Network panel, confirm that the signed-in `GET /api/v1/auths/` request returns HTTP 200 and the session survives the reload.
