/**
 * Jest global setup — Gemma Compute Monitor frontend.
 *
 * This file is executed once per test file, AFTER the test framework has been
 * installed into the environment. It is wired in through `jest.config.js`:
 *
 *     setupFilesAfterEnv: ['<rootDir>/jest.setup.js']
 *
 * Because the project's Jest config is produced by `next/jest`, this file is
 * transformed by Next.js's SWC pipeline, so modern ECMAScript `import` syntax
 * is fully supported here (no `require`/CommonJS needed).
 *
 * Responsibilities (kept intentionally minimal and fast):
 *   1. Register @testing-library/jest-dom's custom DOM matchers globally.
 *   2. Install jest-canvas-mock so <canvas> components mount under jsdom.
 *
 * Keeping this file lean ensures every test file starts quickly and that the
 * mandated frontend tests (auth-gate behavior + the GpuWaveform canvas
 * panel-animation test) have exactly the globals they need — nothing more.
 */

// ---------------------------------------------------------------------------
// 1. @testing-library/jest-dom — custom DOM matchers
// ---------------------------------------------------------------------------
// Importing this package for its side effects extends Jest's global `expect`
// with DOM-aware matchers such as `toBeInTheDocument()`, `toHaveTextContent()`,
// `toBeVisible()`, `toHaveClass()`, `toBeDisabled()`, and `toHaveValue()`.
//
// These matchers are relied upon by the PasswordGate authentication-gate tests
// and the four telemetry-panel tests, allowing assertions to be expressed
// against the rendered jsdom tree in a readable, declarative style.
import '@testing-library/jest-dom';

// ---------------------------------------------------------------------------
// 2. jest-canvas-mock — fake CanvasRenderingContext2D for jsdom
// ---------------------------------------------------------------------------
// jsdom does not implement the HTML Canvas 2D rendering context: by default,
// `HTMLCanvasElement.prototype.getContext('2d')` returns `null`. The
// GpuWaveform panel draws its scrolling GPU-utilization waveform onto a
// <canvas> via `getContext('2d')` inside a `requestAnimationFrame` loop, so
// without a mock it would throw (e.g. "Cannot read properties of null") the
// moment the component mounts in a test.
//
// Importing jest-canvas-mock for its side effects patches the canvas prototype
// so `getContext('2d')` returns a fully stubbed, call-recording 2D context.
// This lets the mandated GpuWaveform panel-animation test render and exercise
// its drawing code under jsdom without a real GPU or native canvas binding.
import 'jest-canvas-mock';

// ---------------------------------------------------------------------------
// Intentionally NOT configured here (documented design decisions)
// ---------------------------------------------------------------------------
// • Network / streaming polyfills (fetch, TextDecoder, ReadableStream):
//   The SSE client hook (useEventSource) consumes a streaming response. Tests
//   that exercise that hook should mock `global.fetch` (and any streaming
//   primitives they touch) WITHIN the specific test file that needs them. We
//   deliberately do not add heavyweight network polyfills to this global setup
//   so the broader suite stays lean and fast; only the test that actually
//   streams pays that cost, and it does so with a precise, purpose-built mock.
//
// • localStorage: jsdom already provides a fully working `localStorage`
//   implementation. The PasswordGate test depends on that real behavior to
//   verify the password is persisted and re-read across renders, so we must
//   NOT replace or stub it here.
