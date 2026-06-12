/// <reference types="@testing-library/jest-dom" />
// The triple-slash reference above loads @testing-library/jest-dom's matcher
// type augmentation (e.g. `toBeInTheDocument`, `toHaveTextContent`) for
// `tsc --noEmit`. At RUNTIME these matchers are registered globally by
// `frontend/jest.setup.js` (`import "@testing-library/jest-dom"`), so no
// runtime import is needed here; this reference is type-only and emits no JS.

/**
 * Auth-gate smoke/integration tests for the Gemma Compute Monitor frontend.
 *
 * Verifies the PasswordGate behaviors mandated by AAP §0.5.3:
 *  - with no / incorrect stored password, only the gate renders (the gated
 *    children / main interface are NOT in the document);
 *  - the correct password persists to localStorage and reveals the children;
 *  - an incorrect password keeps the gate visible and shows the inline error.
 *
 * The gate reads `process.env.NEXT_PUBLIC_PASSWORD` at render time; next/jest
 * does not build-inline NEXT_PUBLIC_* vars, so we set it in beforeEach.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PasswordGate from "@/components/PasswordGate";

const PASSWORD = "secret123";
const STORAGE_KEY = "gcm-auth";

const originalPassword = process.env.NEXT_PUBLIC_PASSWORD;

function renderGate() {
  return render(
    <PasswordGate>
      <div data-testid="protected-child">DASHBOARD CONTENT</div>
    </PasswordGate>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  process.env.NEXT_PUBLIC_PASSWORD = PASSWORD;
});

afterEach(() => {
  window.localStorage.clear();
  if (originalPassword === undefined) {
    delete process.env.NEXT_PUBLIC_PASSWORD;
  } else {
    process.env.NEXT_PUBLIC_PASSWORD = originalPassword;
  }
});

describe("PasswordGate", () => {
  it("renders only the gate (not the protected children) when nothing is stored", async () => {
    renderGate();

    expect(await screen.findByTestId("password-gate")).toBeInTheDocument();
    expect(screen.queryByTestId("protected-child")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Unlock" })).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  it("keeps the gate visible and shows an error on an incorrect password", async () => {
    const user = userEvent.setup();
    renderGate();
    await screen.findByTestId("password-gate");

    await user.type(screen.getByLabelText("Password"), "wrong-password");
    await user.click(screen.getByRole("button", { name: "Unlock" }));

    expect(screen.getByRole("alert")).toHaveTextContent("Incorrect password");
    expect(screen.getByTestId("password-gate")).toBeInTheDocument();
    expect(screen.queryByTestId("protected-child")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("unlocks, persists to localStorage, and renders children on the correct password", async () => {
    const user = userEvent.setup();
    renderGate();
    await screen.findByTestId("password-gate");

    await user.type(screen.getByLabelText("Password"), PASSWORD);
    await user.click(screen.getByRole("button", { name: "Unlock" }));

    expect(await screen.findByTestId("protected-child")).toBeInTheDocument();
    expect(screen.queryByTestId("password-gate")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(PASSWORD);
  });

  it("renders children immediately when a matching value is already stored", async () => {
    window.localStorage.setItem(STORAGE_KEY, PASSWORD);

    renderGate();

    expect(await screen.findByTestId("protected-child")).toBeInTheDocument();
    expect(screen.queryByTestId("password-gate")).not.toBeInTheDocument();
  });

  it("never unlocks when NEXT_PUBLIC_PASSWORD is unset/empty", async () => {
    process.env.NEXT_PUBLIC_PASSWORD = "";
    const user = userEvent.setup();
    renderGate();
    await screen.findByTestId("password-gate");

    // An empty expected password must never grant access, even on empty submit.
    await user.click(screen.getByRole("button", { name: "Unlock" }));

    expect(screen.getByTestId("password-gate")).toBeInTheDocument();
    expect(screen.queryByTestId("protected-child")).not.toBeInTheDocument();
  });
});
