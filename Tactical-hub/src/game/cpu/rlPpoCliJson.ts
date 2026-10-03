export function formatPpoCliJson(result: unknown) {
  return `${JSON.stringify(result, null, 2)}\n`;
}
