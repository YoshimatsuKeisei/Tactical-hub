import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Resolve RL subprocess assets from this module instead of the caller's cwd.
 * This remains stable for npm scripts on Windows and Linux/Colab.
 */
export const RL_PROJECT_ROOT = resolve(fileURLToPath(new URL("../../../", import.meta.url)));
export const RL_VITE_NODE_ENTRY = resolve(RL_PROJECT_ROOT, "node_modules", "vite-node", "vite-node.mjs");
