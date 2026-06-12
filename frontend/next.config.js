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
};

module.exports = nextConfig;
