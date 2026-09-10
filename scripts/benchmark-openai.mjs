#!/usr/bin/env node

const defaults = {
  concurrency: 1,
  maxTokens: 64,
  requests: 3,
  warmup: 1,
};

function usage(message) {
  if (message) console.error(message);
  console.error(
    "usage: node scripts/benchmark-openai.mjs --url URL --model MODEL [--prompt TEXT] [--requests N] [--concurrency N] [--max-tokens N] [--warmup N]",
  );
  process.exitCode = 2;
}

function parsePositiveInteger(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function parseArgs(arguments_) {
  const options = { ...defaults, prompt: "Explain paged key-value cache management." };
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index];
    const value = arguments_[index + 1];
    if (!flag?.startsWith("--") || value === undefined) throw new Error(`invalid argument ${flag}`);
    switch (flag) {
      case "--url": options.url = value; break;
      case "--model": options.model = value; break;
      case "--prompt": options.prompt = value; break;
      case "--requests": options.requests = parsePositiveInteger(value, flag); break;
      case "--concurrency": options.concurrency = parsePositiveInteger(value, flag); break;
      case "--max-tokens": options.maxTokens = parsePositiveInteger(value, flag); break;
      case "--warmup": options.warmup = parsePositiveInteger(value, flag); break;
      default: throw new Error(`unknown flag ${flag}`);
    }
  }
  if (!options.url || !options.model) throw new Error("--url and --model are required");
  return options;
}

async function runRequest(options) {
  const started = performance.now();
  const response = await fetch(`${options.url}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: options.model,
      messages: [{ role: "user", content: options.prompt }],
      temperature: 0,
      max_tokens: options.maxTokens,
      stream: true,
      stream_options: { include_usage: true },
    }),
  });
  if (!response.ok || !response.body) throw new Error(`request failed: ${response.status} ${response.statusText}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  let firstEvent;
  let firstToken;
  let usage;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffered += decoder.decode(value, { stream: true });
    for (;;) {
      const newline = buffered.indexOf("\n");
      if (newline < 0) break;
      const line = buffered.slice(0, newline);
      buffered = buffered.slice(newline + 1);
      if (!line.startsWith("data: ")) continue;
      firstEvent ??= performance.now();
      const data = line.slice(6);
      if (data === "[DONE]") continue;
      const event = JSON.parse(data);
      if (event.choices?.some((choice) => choice.delta?.content)) firstToken ??= performance.now();
      usage ??= event.usage;
      if (event.usage) usage = event.usage;
    }
  }
  const totalMs = performance.now() - started;
  const completionTokens = usage?.completion_tokens ?? null;
  const ttftMs = (firstToken ?? firstEvent ?? performance.now()) - started;
  return {
    ttft_ms: Math.round(ttftMs * 1000) / 1000,
    total_ms: Math.round(totalMs * 1000) / 1000,
    mean_inter_token_latency_ms: completionTokens && completionTokens > 1
      ? Math.round(((totalMs - ttftMs) / (completionTokens - 1)) * 1000) / 1000
      : null,
    completion_tokens: completionTokens,
  };
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  const upper = ordered.length / 2;
  return ordered.length % 2 === 0
    ? (ordered[upper - 1] + ordered[upper]) / 2
    : ordered[Math.floor(upper)];
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  for (let index = 0; index < options.warmup; index += 1) await runRequest(options);

  const started = performance.now();
  const samples = [];
  let next = 0;
  async function worker() {
    while (next < options.requests) {
      next += 1;
      samples.push(await runRequest(options));
    }
  }
  await Promise.all(Array.from({ length: Math.min(options.concurrency, options.requests) }, worker));
  const wallMs = performance.now() - started;
  const completionTokens = samples.reduce((total, sample) => total + (sample.completion_tokens ?? 0), 0);
  const interTokenLatencies = samples
    .map((sample) => sample.mean_inter_token_latency_ms)
    .filter((latency) => latency !== null);
  console.log(JSON.stringify({
    workload: {
      prompt: options.prompt,
      requests: options.requests,
      concurrency: options.concurrency,
      max_tokens: options.maxTokens,
      warmup: options.warmup,
    },
    summary: {
      median_ttft_ms: median(samples.map((sample) => sample.ttft_ms)),
      median_total_ms: median(samples.map((sample) => sample.total_ms)),
      median_mean_inter_token_latency_ms: interTokenLatencies.length > 0
        ? median(interTokenLatencies)
        : null,
      aggregate_completion_tokens_per_s: Math.round((completionTokens / (wallMs / 1000)) * 1000) / 1000,
    },
    samples,
  }, null, 2));
}

main().catch((error) => usage(error.message));
