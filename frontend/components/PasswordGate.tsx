"use client";

/**
 * PasswordGate — single-shared-password access gate for the Gemma Compute
 * Monitor (AAP §0.5.3).
 *
 * Behavior contract:
 *  - On the client (after mount) it compares the value persisted in
 *    `localStorage` against the expected password. If a matching value is
 *    already stored, the gate is bypassed and `children` (the dashboard) is
 *    rendered immediately.
 *  - Until the component has mounted on the client it renders `null`, so the
 *    server-rendered markup and the first client render agree (no React
 *    hydration mismatch). `localStorage` is therefore touched ONLY inside the
 *    mount effect, never during render.
 *  - When locked it renders ONLY the gate (a password input + an "Unlock"
 *    button). Submitting the correct password persists it to `localStorage`
 *    and reveals `children`. Submitting an incorrect password shows an inline
 *    error and keeps the gate visible — `children` is NEVER rendered while
 *    locked.
 *
 * Security note: this is a lightweight shared-access gate, NOT real
 * authentication (multi-user sessions and backend auth are explicitly out of
 * scope per AAP §0.6.2).
 *
 * Configuration: the expected password comes from
 * `process.env.NEXT_PUBLIC_PASSWORD`. Next.js inlines `NEXT_PUBLIC_*` variables
 * into the client bundle at BUILD time, so they are visible in the browser and
 * are NOT secret. Because the value is baked in at build time, changing
 * `NEXT_PUBLIC_PASSWORD` on Railway requires a rebuild/redeploy to take effect.
 * An unset or empty expected password is treated as "never unlock".
 *
 * This is a Client Component (`"use client"` above) because it owns
 * interactive state and reads browser-only `localStorage`.
 */

import {
  useEffect,
  useState,
  type ChangeEvent,
  type FormEvent,
  type ReactNode,
} from "react";

// Fixed, dark-only palette (AAP §0.1.2). These are the only brand colors used
// here; neutral text (#e5e7eb) and the error color (#ef4444) are the sole
// additional neutrals. No light mode, no third-party styling library.
const PALETTE = {
  background: "#0f1117",
  panel: "#1e2130",
  border: "#2d3348",
  primary: "#6366f1",
  counter: "#f59e0b",
} as const;

// localStorage key under which the accepted password is persisted.
const STORAGE_KEY = "gcm-auth";

export default function PasswordGate({ children }: { children: ReactNode }) {
  // Read at render time. Next.js inlines NEXT_PUBLIC_* at build time; an unset
  // value collapses to "" which is treated as "never unlock" below.
  const expected = process.env.NEXT_PUBLIC_PASSWORD ?? "";

  const [mounted, setMounted] = useState(false);
  const [authed, setAuthed] = useState(false);
  const [value, setValue] = useState("");
  const [error, setError] = useState(false);

  useEffect(() => {
    // Mark mounted so the first client render can diverge from SSR output.
    setMounted(true);
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored !== null && expected !== "" && stored === expected) {
        setAuthed(true);
      }
    } catch {
      /* localStorage unavailable (private mode / SSR) -> stay locked */
    }
  }, [expected]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (expected !== "" && value === expected) {
      try {
        window.localStorage.setItem(STORAGE_KEY, value);
      } catch {
        /* ignore persistence failure; still unlock for this session */
      }
      setAuthed(true);
      setError(false);
    } else {
      setError(true);
      setAuthed(false);
    }
  };

  if (!mounted) {
    return null;
  }

  if (authed) {
    return <>{children}</>;
  }

  return (
    <div
      data-testid="password-gate"
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: PALETTE.background,
        color: "#e5e7eb",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          background: PALETTE.panel,
          border: `1px solid ${PALETTE.border}`,
          borderRadius: 12,
          padding: 32,
          width: 360,
          maxWidth: "90vw",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 20, color: PALETTE.counter }}>
          Gemma Compute Monitor
        </h1>
        <label htmlFor="gcm-password" style={{ fontSize: 13, opacity: 0.8 }}>
          Enter access password
        </label>
        <input
          id="gcm-password"
          name="password"
          type="password"
          aria-label="Password"
          placeholder="Password"
          value={value}
          autoComplete="current-password"
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            setValue(e.target.value);
            if (error) setError(false);
          }}
          style={{
            padding: "10px 12px",
            borderRadius: 8,
            border: `1px solid ${PALETTE.border}`,
            background: PALETTE.background,
            color: "#e5e7eb",
            fontSize: 14,
            outline: "none",
          }}
        />
        {error ? (
          <p role="alert" style={{ margin: 0, color: "#ef4444", fontSize: 13 }}>
            Incorrect password
          </p>
        ) : null}
        <button
          type="submit"
          style={{
            padding: "10px 12px",
            borderRadius: 8,
            border: "none",
            background: PALETTE.primary,
            color: "#ffffff",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Unlock
        </button>
      </form>
    </div>
  );
}
