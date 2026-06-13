/**
 * Next.js configuration — Gemma Compute Monitor frontend (App Router, Next.js 16).
 *
 * Intentionally MINIMAL. Next.js 16 builds with Turbopack by default, so this
 * project ships zero bundler customization:
 *   - No custom `webpack(config)` override — that would force the legacy
 *     `--webpack` path and break the default Turbopack `next build`.
 *   - No `experimental` flags, image loaders, or i18n — none are required.
 *   - No `rewrites`/`redirects`/proxy — the dashboard talks to the FastAPI
 *     backend cross-origin over the ngrok HTTPS URL in `NEXT_PUBLIC_API_URL`;
 *     CORS is enforced on the backend (allow-list includes the Railway origin),
 *     so no Next.js rewrite is needed here.
 *
 * The UI is dark-only via CSS, so there is no theme toggle configured here.
 *
 * Railway note: the deploy runs `npm run build` (→ `next build`) then
 * `npm run start` (→ `next start`). `output: "standalone"` is deliberately
 * omitted because plain `next start` does not consume the standalone server
 * bundle, and keeping this config minimal reduces the ways the mandatory
 * Railway build can fail.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  // Enable React Strict Mode. In development this double-invokes effects and
  // their cleanup functions, surfacing lifecycle bugs in the SSE `EventSource`
  // hook and the `requestAnimationFrame`-driven canvas waveform (i.e. it
  // verifies that subscriptions and animation frames are torn down correctly
  // on unmount). It is a no-op in production builds.
  reactStrictMode: true,

  // Do not advertise the framework via the `X-Powered-By: Next.js` response
  // header (QA FINAL-ACCEPTANCE Issue 7 — framework fingerprinting). `next
  // start` (the Railway production server) omits the header entirely with this
  // set to false.
  poweredByHeader: false,

  /**
   * Security/hardening response headers applied to every route (QA
   * FINAL-ACCEPTANCE Issue 7). These complement — they do not replace — the
   * backend's CORS allow-list and its own response-header middleware.
   *
   * @returns {Promise<import('next').NextConfig['headers']>}
   */
  async headers() {
    // Content-Security-Policy tuned for this app:
    //   - 'unsafe-inline' (style + script) and 'unsafe-eval' (script) are
    //     required because Next.js injects an inline bootstrap/hydration
    //     <script>/<style>; omitting them would break the app shell and, in
    //     development, Fast Refresh.
    //   - connect-src allows https:/http: so the dashboard's EventSource and
    //     /health fetch reach the cross-origin FastAPI backend over the dynamic
    //     ngrok HTTPS URL (NEXT_PUBLIC_API_URL), and ws:/wss: so the dev-server
    //     HMR socket is not blocked. The backend host is not known at build
    //     time, so it cannot be pinned more narrowly here.
    //   - object-src 'none', frame-ancestors 'none', and base-uri 'self' keep
    //     the strong, non-negotiable protections regardless of the above.
    const contentSecurityPolicy = [
      "default-src 'self'",
      "base-uri 'self'",
      "object-src 'none'",
      "frame-ancestors 'none'",
      "form-action 'self'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "style-src 'self' 'unsafe-inline'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
      "connect-src 'self' https: http: ws: wss:",
    ].join("; ");

    const securityHeaders = [
      { key: "Content-Security-Policy", value: contentSecurityPolicy },
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "X-Frame-Options", value: "DENY" },
      { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
    ];

    // HSTS is emitted ONLY for the deployed (HTTPS, Railway) production server,
    // never for local plain-HTTP `next dev`. Browsers ignore an HSTS header
    // received over HTTP anyway (RFC 6797 §8.1), but gating it to production
    // keeps the development environment clean.
    if (process.env.NODE_ENV === "production") {
      securityHeaders.push({
        key: "Strict-Transport-Security",
        value: "max-age=63072000; includeSubDomains",
      });
    }

    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

module.exports = nextConfig;
