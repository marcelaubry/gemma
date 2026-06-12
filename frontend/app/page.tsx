/**
 * Route `/` entry page for the Gemma Compute Monitor (AAP §0.5.3).
 *
 * This is intentionally a **Server Component** (note the deliberate absence
 * of the client directive at the top of this file). It is a thin wrapper that
 * composes the access gate around the dashboard:
 *
 *     <PasswordGate>
 *       <Dashboard />
 *     </PasswordGate>
 *
 * Why this stays a Server Component (canonical App Router "interleaving"):
 *   - Both `PasswordGate` and `Dashboard` are Client Components.
 *   - A Server Component MAY render Client Components, and it MAY pass a client
 *     element (`<Dashboard />`) as the `children` prop of another Client
 *     Component (`<PasswordGate>`). React only mounts the dashboard's client
 *     logic once `PasswordGate` unlocks and renders `{children}`.
 *   - Keeping this page server-rendered avoids turning the whole route into a
 *     client bundle — only the gate and dashboard ship as client components.
 *
 * The gate WRAPS the dashboard so the dashboard is rendered only after the
 * correct password is entered; `Dashboard` itself never imports `PasswordGate`,
 * by design — the composition lives here.
 *
 * The document shell (`<html>`/`<body>`, metadata) lives in `layout.tsx` and
 * the theme in `globals.css`, so this file deliberately holds no additional
 * markup, providers, styling, or client state.
 */

import PasswordGate from "@/components/PasswordGate";
import Dashboard from "@/components/Dashboard";

export default function Home() {
  return (
    <PasswordGate>
      <Dashboard />
    </PasswordGate>
  );
}
