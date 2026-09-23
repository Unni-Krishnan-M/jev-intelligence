/**
 * Post-login redirect targets (`?next=`). Only same-origin paths are accepted.
 *
 * A prefix check ("starts with / but not //") is not enough: browsers treat "\" as "/" and drop tabs
 * and newlines while parsing, so "/\evil.com" and "/<TAB>/evil.com" both resolve to http://evil.com/.
 * The value is resolved against our own origin and accepted only if the origin is unchanged; values
 * containing backslashes or control characters (raw or percent-encoded) are rejected outright.
 */
const UNSAFE = /[\\\u0000-\u001f\u007f]/;

function decoded(value: string): string | null {
  try {
    return decodeURIComponent(value);
  } catch {
    return null; // malformed percent-encoding
  }
}

export function safeNextPath(next: string | null | undefined, origin: string): string | null {
  if (!next || next.length > 2048) return null;
  const plain = decoded(next);
  if (plain === null) return null;
  for (const v of [next, plain]) {
    if (UNSAFE.test(v) || !v.startsWith("/") || v.startsWith("//")) return null;
  }
  let base: URL;
  let url: URL;
  try {
    base = new URL(origin);
    url = new URL(next, base);
  } catch {
    return null;
  }
  if (url.origin !== base.origin) return null;
  return url.pathname + url.search + url.hash;
}
