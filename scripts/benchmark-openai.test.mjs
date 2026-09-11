import assert from "node:assert/strict";
import { once } from "node:events";
import http from "node:http";
import { spawn } from "node:child_process";
import test from "node:test";

const script = new URL("./benchmark-openai.mjs", import.meta.url);

async function withServer(responder, run) {
  const server = http.createServer(responder);
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const { port } = server.address();
  try {
    return await run(`http://127.0.0.1:${port}`);
  } finally {
    server.close();
    await once(server, "close");
  }
}

function runCli(url, extra = []) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script.pathname,
      "--url", url,
      "--model", "mock-model",
      "--cache-condition", "warm",
      "--warmup", "0",
      "--requests", "1",
      "--concurrency", "1",
      "--max-tokens", "2",
      "--seed", "7",
      "--prompt", "a reproducible prompt",
      ...extra,
    ], { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
  });
}

async function stream(res, fragments) {
  res.writeHead(200, { "content-type": "text/event-stream" });
  for (const fragment of fragments) {
    res.write(fragment);
    await new Promise(setImmediate);
  }
  res.end();
}

test("records fragmented CRLF SSE content and reported usage", async () => {
  let receivedPayload;
  await withServer(async (request, response) => {
    assert.equal(request.url, "/v1/chat/completions");
    let body = "";
    for await (const chunk of request) body += chunk;
    receivedPayload = JSON.parse(body);
    const payload = [
      ": keepalive\r\n",
      "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\r\n\r\n",
      "data: {\"choices\":[{\"delta\":{\"content\":\"hel",
      "lo\"}}]}\r\n\r\n",
      "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\r\n\r\n",
      "data: {\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\r\n\r\n",
      "data: [DONE]\r\n\r\n",
    ];
    await stream(response, payload);
  }, async (url) => {
    const result = await runCli(url);
    assert.equal(result.code, 0, result.stderr);
    const output = JSON.parse(result.stdout);
    assert.equal(output.schema_version, 2);
    assert.equal(output.workload.prompt_sha256.length, 64);
    assert.equal("prompt" in output.workload, false);
    assert.equal(output.workload.timeout_ms, 120_000);
    assert.deepEqual(output.workload.sampling, { temperature: 0, seed: 7 });
    assert.equal(output.samples[0].received_done, true);
    assert.equal(output.samples[0].content_delta_count, 2);
    assert.equal(output.samples[0].prompt_tokens, 4);
    assert.equal(output.samples[0].completion_tokens, 2);
    assert.notEqual(output.samples[0].ttft_ms, null);
    assert.equal(output.summary.reported_completion_tokens.total, 2);
    assert.equal(output.summary.aggregate_reported_completion_tokens_per_s !== null, true);
  });
  assert.deepEqual(receivedPayload, {
    model: "mock-model",
    messages: [{ role: "user", content: "a reproducible prompt" }],
    temperature: 0,
    max_tokens: 2,
    stream: true,
    stream_options: { include_usage: true },
    seed: 7,
  });
});

test("rejects an incomplete SSE stream instead of emitting a benchmark sample", async () => {
  await withServer(async (_request, response) => {
    await stream(response, ["data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\r\n\r\n"]);
  }, async (url) => {
    const result = await runCli(url);
    assert.notEqual(result.code, 0);
    assert.equal(result.stdout, "");
    assert.match(result.stderr, /stream ended without \[DONE\]/);
  });
});

test("rejects an in-band SSE error event", async () => {
  await withServer(async (_request, response) => {
    await stream(response, ["data: {\"error\":{\"message\":\"model unavailable\"}}\r\n\r\n", "data: [DONE]\r\n\r\n"]);
  }, async (url) => {
    const result = await runCli(url);
    assert.notEqual(result.code, 0);
    assert.equal(result.stdout, "");
    assert.match(result.stderr, /server returned an SSE error event/);
  });
});

test("rejects invalid negative token usage", async () => {
  await withServer(async (_request, response) => {
    await stream(response, ["data: {\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":-1}}\r\n\r\n", "data: [DONE]\r\n\r\n"]);
  }, async (url) => {
    const result = await runCli(url);
    assert.notEqual(result.code, 0);
    assert.equal(result.stdout, "");
    assert.match(result.stderr, /invalid completion_tokens usage value/);
  });
});

test("cancels the HTTP reader once DONE arrives", async () => {
  let requestClosed = false;
  await withServer(async (request, response) => {
    request.on("error", () => {});
    response.on("error", () => {});
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write("data: [DONE]\r\n\r\n");
    await new Promise((resolve) => request.once("close", resolve));
    requestClosed = true;
    response.end();
  }, async (url) => {
    const result = await runCli(url, ["--timeout-ms", "250"]);
    assert.equal(result.code, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).samples[0].received_done, true);
  });
  assert.equal(requestClosed, true);
});

test("times out a never-ending stream", async () => {
  await withServer((_request, response) => {
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write(": keepalive\r\n");
  }, async (url) => {
    const result = await runCli(url, ["--timeout-ms", "30"]);
    assert.notEqual(result.code, 0);
    assert.equal(result.stdout, "");
    assert.match(result.stderr, /request timed out after 30 ms/);
  });
});

test("rejects a cold-start declaration with warmup requests", async () => {
  const result = await runCli("http://127.0.0.1:1", ["--cache-condition", "cold-start", "--warmup", "1"]);
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /cold-start requires --warmup 0/);
});
