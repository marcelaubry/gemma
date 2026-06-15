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

// All styling lives in `frontend/app/globals.css` under the `.gcm-gate*`
// classes (dark-only, built entirely from the palette CSS custom properties).
// Keeping styles there — rather than as inline React `style` objects — removes
// duplicated palette values from this component and lets the gate express
// keyboard-focus, hover, active, and disabled states, which inline style
// objects cannot represent.

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
    <div data-testid="password-gate" className="gcm-gate">
      <form onSubmit={handleSubmit} className="gcm-gate__card">
        <h1 className="gcm-gate__title">Gemma Compute Monitor</h1>
        <label htmlFor="gcm-password" className="gcm-gate__label">
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
          className="gcm-gate__input"
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            setValue(e.target.value);
            if (error) setError(false);
          }}
        />
        {error ? (
          <p role="alert" className="gcm-gate__error">
            Incorrect password
          </p>
        ) : null}
        <button type="submit" className="gcm-gate__button">
          Unlock
        </button>
      </form>
    </div>
  );
}
