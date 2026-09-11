#!/usr/bin/env node

import { createHash } from "node:crypto";

const defaults = {
  concurrency: 1,
  maxTokens: 64,
  requests: 3,
  timeoutMs: 120_000,
  warmup: 1,
};

const cacheConditions = new Set(["cold-start", "warm", "mixed"]);

function usage(message) {
  if (message) console.error(message);
  console.error(
    "usage: node scripts/benchmark-openai.mjs --url URL --model MODEL --cache-condition cold-start|warm|mixed [--prompt TEXT] [--requests N] [--concurrency N] [--max-tokens N] [--warmup N] [--seed N] [--timeout-ms N]",
  );
  process.exitCode = 2;
}

function parsePositiveInteger(value, name) {
  const parsed = Number(value);
  if (!/^[1-9][0-9]*$/.test(value) || !Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInteger(value, name) {
  const parsed = Number(value);
  if (!/^(?:0|[1-9][0-9]*)$/.test(value) || !Number.isSafeInteger(parsed)) {
    throw new Error(`${name} must be a non-negative integer`);
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
      case "--url": options.url = value.replace(/\/+$/, ""); break;
      case "--model": options.model = value; break;
      case "--cache-condition": options.cacheCondition = value; break;
      case "--prompt": options.prompt = value; break;
      case "--requests": options.requests = parsePositiveInteger(value, flag); break;
      case "--concurrency": options.concurrency = parsePositiveInteger(value, flag); break;
      case "--max-tokens": options.maxTokens = parsePositiveInteger(value, flag); break;
      case "--timeout-ms": options.timeoutMs = parsePositiveInteger(value, flag); break;
      case "--warmup": options.warmup = parseNonNegativeInteger(value, flag); break;
      case "--seed": {
        const seed = Number(value);
        if (!/^(?:0|[1-9][0-9]*)$/.test(value) || !Number.isSafeInteger(seed)) {
          throw new Error("--seed must be a non-negative integer");
        }
        options.seed = seed;
        break;
      }
      default: throw new Error(`unknown flag ${flag}`);
    }
  }
  if (!options.url || !options.model) throw new Error("--url and --model are required");
  if (!cacheConditions.has(options.cacheCondition)) {
    throw new Error("--cache-condition must be one of cold-start, warm, or mixed");
  }
  if (options.cacheCondition === "cold-start" && options.warmup > 0) {
    throw new Error("--cache-condition cold-start requires --warmup 0");
  }
  return options;
}

function createSseParser(onEvent) {
  let buffered = "";
  let dataLines = [];

  function dispatch() {
    if (dataLines.length > 0) onEvent(dataLines.join("\n"));
    dataLines = [];
  }

  function processLine(line) {
    if (line === "") return dispatch();
    if (line.startsWith(":")) return;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") dataLines.push(value);
  }

  return {
    push(text) {
      buffered += text;
      for (;;) {
        const newline = buffered.indexOf("\n");
        if (newline < 0) return;
        let line = buffered.slice(0, newline);
        buffered = buffered.slice(newline + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        processLine(line);
      }
    },
    finish() {
      if (buffered.length > 0) processLine(buffered.replace(/\r$/, ""));
      dispatch();
    },
  };
}

async function runRequest(options) {
  const started = performance.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs);
  const payload = {
    model: options.model,
    messages: [{ role: "user", content: options.prompt }],
    temperature: 0,
    max_tokens: options.maxTokens,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (options.seed !== undefined) payload.seed = options.seed;
  let reader;
  try {
    const response = await fetch(`${options.url}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) throw new Error(`request failed: ${response.status} ${response.statusText}`);

    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let firstEvent;
    let firstContentDelta;
    let lastContentDelta;
    let contentDeltaCount = 0;
    let usage;
    let receivedDone = false;
    const parser = createSseParser((data) => {
    const now = performance.now();
    firstEvent ??= now;
    if (data === "[DONE]") {
      receivedDone = true;
      return;
    }
    let event;
    try {
      event = JSON.parse(data);
    } catch {
      throw new Error("server returned an invalid JSON SSE data event");
    }
      if (event.error) throw new Error("server returned an SSE error event");
    if (event.choices?.some((choice) => {
      const content = choice.delta?.content;
      return typeof content === "string" ? content.length > 0 : Array.isArray(content) && content.length > 0;
    })) {
      firstContentDelta ??= now;
      lastContentDelta = now;
      contentDeltaCount += 1;
    }
      if (event.usage) {
        tokenCount(event.usage.completion_tokens, "completion_tokens");
        tokenCount(event.usage.prompt_tokens, "prompt_tokens");
        usage = event.usage;
      }
    });
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
      if (receivedDone) {
        await reader.cancel();
        break;
      }
    }
    parser.push(decoder.decode());
    parser.finish();
    if (!receivedDone) throw new Error("stream ended without [DONE]");
    const totalMs = performance.now() - started;
    const completionTokens = tokenCount(usage?.completion_tokens, "completion_tokens");
    const promptTokens = tokenCount(usage?.prompt_tokens, "prompt_tokens");
    const firstStreamEventMs = firstEvent === undefined ? null : firstEvent - started;
    const ttftMs = firstContentDelta === undefined ? null : firstContentDelta - started;
    return {
    first_stream_event_ms: firstStreamEventMs === null ? null : Math.round(firstStreamEventMs * 1000) / 1000,
    ttft_ms: ttftMs === null ? null : Math.round(ttftMs * 1000) / 1000,
    total_ms: Math.round(totalMs * 1000) / 1000,
    mean_inter_content_delta_latency_ms: contentDeltaCount > 1
      ? Math.round(((lastContentDelta - firstContentDelta) / (contentDeltaCount - 1)) * 1000) / 1000
      : null,
    estimated_mean_inter_token_latency_ms: completionTokens && completionTokens > 1 && firstContentDelta !== undefined
      ? Math.round(((lastContentDelta - firstContentDelta) / (completionTokens - 1)) * 1000) / 1000
      : null,
    content_delta_count: contentDeltaCount,
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
      received_done: receivedDone,
    };
  } catch (error) {
    await reader?.cancel(error).catch(() => {});
    if (controller.signal.aborted) throw new Error(`request timed out after ${options.timeoutMs} ms`);
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function tokenCount(value, name) {
  if (value === undefined) return null;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`server returned an invalid ${name} usage value`);
  }
  return value;
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  const upper = ordered.length / 2;
  return ordered.length % 2 === 0
    ? (ordered[upper - 1] + ordered[upper]) / 2
    : ordered[Math.floor(upper)];
}

function medianPresent(samples, field) {
  const values = samples.map((sample) => sample[field]).filter((value) => value !== null);
  return { sample_count: values.length, value: values.length > 0 ? median(values) : null };
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
  const samplesWithCompletionUsage = samples.filter((sample) => sample.completion_tokens !== null);
  const completionTokens = samplesWithCompletionUsage
    .reduce((total, sample) => total + sample.completion_tokens, 0);
  const promptDigest = createHash("sha256").update(options.prompt, "utf8").digest("hex");
  console.log(JSON.stringify({
    schema_version: 2,
    workload: {
      endpoint: options.url,
      model: options.model,
      prompt_sha256: promptDigest,
      prompt_utf8_bytes: Buffer.byteLength(options.prompt, "utf8"),
      requests: options.requests,
      concurrency: options.concurrency,
      max_tokens: options.maxTokens,
      timeout_ms: options.timeoutMs,
      warmup_requests_excluded: options.warmup,
      sampling: { temperature: 0, seed: options.seed ?? null },
      streaming: { requested: true, include_usage: true },
      cache_condition: {
        declared: options.cacheCondition,
        note: "Operator-declared: this client neither clears nor inspects server caches.",
      },
    },
    summary: {
      measured_wall_ms: Math.round(wallMs * 1000) / 1000,
      median_ttft_ms: medianPresent(samples, "ttft_ms"),
      median_first_stream_event_ms: medianPresent(samples, "first_stream_event_ms"),
      median_total_ms: medianPresent(samples, "total_ms"),
      median_mean_inter_content_delta_latency_ms: medianPresent(samples, "mean_inter_content_delta_latency_ms"),
      median_estimated_mean_inter_token_latency_ms: medianPresent(samples, "estimated_mean_inter_token_latency_ms"),
      reported_completion_tokens: {
        total: completionTokens,
        sample_count: samplesWithCompletionUsage.length,
      },
      aggregate_reported_completion_tokens_per_s: samplesWithCompletionUsage.length === samples.length
        ? Math.round((completionTokens / (wallMs / 1000)) * 1000) / 1000
        : null,
    },
    samples,
  }, null, 2));
}

main().catch((error) => usage(error.message));
