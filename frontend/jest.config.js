/**
 * Jest configuration — Gemma Compute Monitor frontend (Next.js 16 + React 19).
 *
 * This file wires up the chosen frontend test setup (Jest + React Testing
 * Library) that runs the mandated smoke/integration tests under
 * `frontend/__tests__/` — the PasswordGate authentication-gate behavior and the
 * telemetry-panel render/animation tests (including the canvas-based
 * GpuWaveform panel).
 *
 * It is produced with **`next/jest`**, the officially documented Next.js way to
 * configure Jest. `next/jest` does the heavy lifting that would otherwise be
 * brittle to hand-roll:
 *   - Transforms `.ts` / `.tsx` (and `.js` / `.jsx`) using Next.js's built-in
 *     SWC compiler — the SAME transform used by `next build` / `next dev` — so
 *     test code and application code compile identically. No Babel or `ts-jest`
 *     configuration is required (and adding one would conflict; see below).
 *   - Stubs out CSS Modules, global stylesheets, and static asset imports so
 *     components that `import './globals.css'` (or similar) mount cleanly under
 *     jsdom instead of throwing on an unparseable import.
 *   - Loads `next.config.js` and the project's `.env*` files into the test
 *     environment (driven by the `dir` option below).
 *
 * This is an executable **CommonJS** module (`.js`, using `require` /
 * `module.exports`), which is what Jest expects for a `jest.config.js` file.
 *
 * @see https://nextjs.org/docs/app/guides/testing/jest
 */

// `next/jest` exposes a factory that returns a function which, given a custom
// Jest config, produces the final (async) Jest configuration with Next.js's
// SWC transform and asset/CSS mocks already merged in.
const nextJest = require('next/jest');

// `dir` points at the root of the Next.js application (this `frontend/`
// directory, where `next.config.js` and the `.env*` files live). `next/jest`
// uses it to load `next.config.js` and the environment files into the test
// environment so tests run against the same configuration as the real app.
const createJestConfig = nextJest({
  dir: './',
});

/**
 * Project-specific Jest options. `next/jest` merges these on top of the
 * Next.js-managed defaults (the SWC `transform`, CSS/asset `moduleNameMapper`
 * entries, `transformIgnorePatterns`, etc.). Keep this object focused on the
 * concerns that are unique to this test suite.
 *
 * @type {import('jest').Config}
 */
const customJestConfig = {
  // Run `jest.setup.js` once per test file, AFTER the test framework is
  // installed. That file registers @testing-library/jest-dom's custom DOM
  // matchers (e.g. `toBeInTheDocument`) and installs `jest-canvas-mock` so the
  // GpuWaveform panel's `getContext('2d')` call works under jsdom.
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],

  // Use a jsdom DOM environment so React components can render and be queried.
  // This backs the PasswordGate render/`localStorage` assertions and the panel
  // render/update tests. (jsdom is provided by the `jest-environment-jsdom`
  // dev dependency.)
  testEnvironment: 'jest-environment-jsdom',

  // Map the `@/…` path alias to the project root, MIRRORING the `paths` alias
  // declared in `tsconfig.json` (`"@/*": ["./*"]` with `baseUrl: "."`). This
  // keeps test imports such as `@/components/PasswordGate` and `@/lib/types`
  // resolving to the same files Next.js/TypeScript resolve them to. `<rootDir>`
  // is this `frontend/` directory (the location of this config file).
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },

  // Only treat files under any `__tests__/` directory whose name ends in
  // `.test.ts`, `.test.tsx`, `.test.js`, or `.test.jsx` as test files. This
  // scopes the runner to the intended `frontend/__tests__/**` suite and avoids
  // accidentally picking up application source modules.
  testMatch: ['**/__tests__/**/*.test.[jt]s?(x)'],
};

// `createJestConfig` returns an async config factory (it loads `next.config.js`
// asynchronously). Exporting it directly — rather than awaiting it here — is
// the documented pattern: Jest resolves the returned promise/function itself,
// which guarantees the Next.js SWC transform is fully wired before tests run.
module.exports = createJestConfig(customJestConfig);
