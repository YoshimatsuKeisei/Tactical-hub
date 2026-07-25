const readline = require("node:readline");

const mode = process.argv[2] ?? "valid";
const lines = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

lines.on("line", (line) => {
  const request = JSON.parse(line);
  if (request.type === "init") {
    process.stdout.write(`${JSON.stringify({ type: "ready", schemaVersion: request.featureSpec.schemaVersion })}\n`);
    return;
  }
  if (request.type === "act") {
    const actionCount = request.actions.length;
    const actionIndex = mode === "invalid" ? actionCount : Math.max(0, actionCount - 1);
    process.stdout.write(`${JSON.stringify({
      type: "action",
      requestId: request.requestId,
      actionIndex,
      value: 0.25,
    })}\n`);
    return;
  }
  if (request.type === "close") {
    process.stdout.write(`${JSON.stringify({ type: "closed" })}\n`);
    lines.close();
  }
});
