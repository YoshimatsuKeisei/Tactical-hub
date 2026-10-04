export class LocalGameSaveChecksumUnavailableError extends Error {
  constructor() {
    super("Web Crypto SHA-256 is unavailable");
    this.name = "LocalGameSaveChecksumUnavailableError";
  }
}

export async function calculateLocalGameSaveSha256(serializedSave: string, cryptoProvider: Crypto | null | undefined = globalThis.crypto): Promise<string> {
  if (!cryptoProvider?.subtle) throw new LocalGameSaveChecksumUnavailableError();
  const bytes = new TextEncoder().encode(serializedSave);
  const digest = await cryptoProvider.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
