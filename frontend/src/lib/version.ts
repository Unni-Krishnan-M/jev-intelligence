/** The web app's version, shown in the footer. Keep in step with package.json (a unit test checks). */
export const APP_VERSION = "1.3.0";

/** "1.3.0" → "v1.3": the footer shows major.minor only. */
export function shortVersion(v: string = APP_VERSION): string {
  const [major, minor] = v.split(".");
  return `v${major}.${minor ?? "0"}`;
}
