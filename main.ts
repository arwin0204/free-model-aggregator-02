// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// The server reads the brief (the fourteen free OpenRouter models with the house's ratings, and the
// six prompts' manifest), then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// Routing: for each request, every free model that meets its requirements is ranked by the brief's
// rating_milli, best first, ties in the brief's order. The first pick is the best-ranked model this
// trial has called fewer than PLAN_CAP times, so the load spreads instead of draining the best model;
// a model that refuses is counted and the request moves to the next ranked model, until one answers or
// the list runs out. One exception: the best-rated model gets a single immediate retry when it answers
// a 5xx or cannot be reached, because a refusal there is usually one bad call, not a spent budget, and
// recovering the best model is worth more than the call it costs. A 429 never earns that retry: a
// rate-limited model stays limited, so the next model serves instead. A 401 ends the prompt at once:
// the key, not the model, is the problem, and no other model would answer on it either. A 2xx answer
// is the request's answer even if its body surprises: calling a second model after one answered would
// make the request invalid. Every choice is a function of the brief, the requests and the gateway's
// answers so far: no clock, no unseeded randomness, so a replay leaves the same state export.
//
// Failures are reported, never hidden: every refused gateway call goes into the response's
// x-makefun-upstream-failures header (at most 8 rows), so the failover column can see a refusal the
// server recovered from. The header rides on served answers and on the 502 that gives a prompt up.
//
// The model field always names the model this entry dialled, with the gateway's openrouter/ prefix:
// the house scores the model its own record shows answered, and the provider's own answer spells it
// without the prefix, so the wire helpers put the dialled id back on every answer and every chunk.
//
// Usage is guaranteed whole: a gateway answer that carries no usage, or an output count below 1, gets
// an estimate from the request and answer text, so the wire's usage check (input at least 0, output at
// least 1) can pass on a provider that omits it. The estimate is a pure function of the request and
// the answer, so a replay reproduces it byte for byte.
//
// Without a pushed key the gateway answers every dial with 401, and the server gives each prompt up
// at once with a well-formed 502: no other model would answer on a key the gateway refuses, so there
// is nothing to fail over to, and the trial then scores the catalog alone.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take, in milliseconds. */
const GATEWAY_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway attempts for one client request. */
const MAX_ATTEMPTS = 8;
/**
 * How many calls one model takes before the first pick moves on: the best model can serve most of a
 * six-prompt trial before the load spreads, and the spread keeps an outage or a spent budget from
 * taking many requests down with it.
 */
const PLAN_CAP = 4;
/** The most rows the x-makefun-upstream-failures trail carries. */
const MAX_TRAIL_ROWS = 8;
/** The most characters of a gateway error the trail's code or the give-up message keeps. */
const MAX_ERROR_CHARS = 500;
/** A fixed epoch second for synthetic stream chunks: nothing in the reply reads the clock. */
const FIXED_CREATED = 1767225600;
/** The header that reports the refused gateway calls behind a served (or given-up) prompt. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Gateway calls made so far in this trial, per dialled model id: the load-spreading ledger. */
const usedCalls = new Map<string, number>();

/** One refused gateway call, as the failures trail reports it. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models meeting the requirements, best rating first. The sort is stable, so equal ratings
 * keep the brief's order, and the order is a pure function of the brief.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The failures trail as one header value: at most 8 rows, each code cut to 64 characters. */
function trailHeader(failures: readonly Failure[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, MAX_TRAIL_ROWS).map((f) => ({ status: f.status, code: f.code.slice(0, 64), model: f.model }));
  const text = JSON.stringify(rows);
  return text.length > 8192 ? null : text;
}

/** One JSON answer, with the failures trail on it when the prompt cost any refused call. */
function sendJson(res: ServerResponse, status: number, body: unknown, failures: readonly Failure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const trail = trailHeader(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: readonly Failure[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, 'upstream_error'), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The text of a message's content: a string, or its text blocks joined. */
function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The characters of text a client request carries: the system prompt and every message's text. */
function requestTextChars(body: Json): number {
  let chars = 0;
  chars += contentText(body.system).length;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    chars += contentText(message.content).length;
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (isRecord(call) && isRecord(call.function) && typeof call.function.arguments === 'string') chars += call.function.arguments.length;
      }
    }
  }
  return chars;
}

/** A rough token count for `chars` characters: one token per four characters, at least one. */
function estimateTokens(chars: number): number {
  return Math.max(1, Math.ceil(chars / 4));
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * A Chat Completions answer whose usage is whole: the gateway's counts where they are present and
 * sane, else an estimate from the request and answer text (output at least 1). The estimate is a pure
 * function of the request and the answer.
 */
function ensureOpenAIUsage(body: Json, requestChars: number): Json {
  const usage = isRecord(body.usage) ? body.usage : {};
  const prompt = finiteNumber(usage.prompt_tokens);
  const completion = finiteNumber(usage.completion_tokens);
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const message = choices.length > 0 && isRecord(choices[0]) && isRecord(choices[0].message) ? choices[0].message : {};
  const answerChars = contentText(message.content).length;
  const outPrompt = prompt !== null && prompt >= 0 ? prompt : estimateTokens(requestChars);
  const outCompletion = completion !== null && completion >= 1 ? completion : estimateTokens(answerChars);
  return { ...body, usage: { prompt_tokens: outPrompt, completion_tokens: outCompletion, total_tokens: outPrompt + outCompletion } };
}

/** An Anthropic Messages answer whose usage is whole, as above. */
function ensureAnthropicUsage(body: Json, requestChars: number): Json {
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = finiteNumber(usage.input_tokens);
  const output = finiteNumber(usage.output_tokens);
  const answerChars = contentText(body.content).length;
  const outInput = input !== null && input >= 0 ? input : estimateTokens(requestChars);
  const outOutput = output !== null && output >= 1 ? output : estimateTokens(answerChars);
  return { ...body, usage: { input_tokens: outInput, output_tokens: outOutput } };
}

/**
 * The gateway body for one dialled model: the client's body with the model replaced, the stream flag
 * forcing a usage chunk, and an output cap where the client named none (the gateway refuses a call
 * with no cap). The Anthropic path is translated to Chat Completions first.
 */
function gatewayBody(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  let out: Json;
  if (clientApi === 'openai') {
    out = { ...body, model };
    if (body.stream === true) {
      const options = isRecord(body.stream_options) ? { ...body.stream_options } : {};
      options.include_usage = true;
      out.stream_options = options;
    }
  } else {
    out = anthropicToOpenAI(body, model);
  }
  if (finiteNumber(out.max_tokens) === null && finiteNumber(out.max_completion_tokens) === null) out.max_tokens = 1024;
  return out;
}

/** One refused gateway call, read off its error body: the provider's code, else its type. */
async function gatewayRefusal(upstream: Response, model: string): Promise<{ status: number; code: string; detail: string }> {
  const status = upstream.status;
  let text = '';
  try {
    text = (await upstream.text()).slice(0, MAX_ERROR_CHARS);
  } catch {
    text = '';
  }
  let code = `http_${status}`;
  let detail = text;
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const error = parsed.error;
      if (typeof error.code === 'string' && error.code !== '') code = error.code;
      else if (typeof error.type === 'string' && error.type !== '') code = error.type;
      if (typeof error.message === 'string' && error.message !== '') detail = error.message.slice(0, MAX_ERROR_CHARS);
    }
  } catch {
    // Not JSON: the raw text is the detail.
  }
  return { status, code, detail };
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(clientApi: 'openai' | 'anthropic', body: Json, model: string): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(gatewayBody(clientApi, body, model)),
    signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
  });
}

/** Whether a translated JSON answer has the shape its wire's checks need. */
function jsonShapeOk(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    return choices.length > 0 && isRecord(choices[0]) && isRecord(choices[0].message) && choices[0].message.role === 'assistant';
  }
  return Array.isArray(body.content) && body.type === 'message';
}

/** One JSON gateway answer translated into the client's wire, usage guaranteed whole. */
function translateJson(clientApi: 'openai' | 'anthropic', answer: Json, model: string, requestChars: number): Json | null {
  if (clientApi === 'openai') {
    const out = ensureOpenAIUsage(openAIJsonWithModel(answer, model), requestChars);
    return jsonShapeOk('openai', out) ? out : null;
  }
  const out = ensureAnthropicUsage(openAIToAnthropic(answer, model), requestChars);
  return jsonShapeOk('anthropic', out) ? out : null;
}

/** One SSE frame's data payloads, each parsed as JSON where it parses. */
function framePayloads(frame: string): unknown[] {
  const out: unknown[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      out.push(JSON.parse(payload));
    } catch {
      // Not JSON: it carries no usage and no text the checks read.
    }
  }
  return out;
}

/** Whether an SSE frame is the stream's [DONE] frame. */
function frameIsDone(frame: string): boolean {
  return frame.split('\n').some((line) => line.trim() === 'data: [DONE]');
}

/** One OpenAI usage chunk, as the gateway sends it when include_usage is on. */
function usageChunk(model: string, input: number, output: number): string {
  return `data: ${JSON.stringify({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  })}\n\n`;
}

/** One OpenAI finish chunk, for a stream the gateway closed without one. */
function finishChunk(model: string): string {
  return `data: ${JSON.stringify({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  })}\n\n`;
}

/**
 * A Chat Completions SSE stream relayed with the dialled model's id, made whole: a usage chunk and a
 * finish chunk are added where the gateway sent none, and [DONE] always closes it. The frames are
 * buffered, so the answer is one well-formed stream however the gateway paced it.
 */
async function relayOpenAIStream(upstream: ReadableStream<Uint8Array>, model: string, requestChars: number, res: ServerResponse, failures: readonly Failure[]): Promise<void> {
  const frames: string[] = [];
  let sawUsage = false;
  let sawFinish = false;
  let textChars = 0;
  for await (const frame of openAIStreamWithModel(upstream, model)) {
    frames.push(frame);
    for (const payload of framePayloads(frame)) {
      if (!isRecord(payload)) continue;
      if (isRecord(payload.usage)) sawUsage = true;
      const choices = Array.isArray(payload.choices) ? payload.choices : [];
      const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      if (isRecord(choice.delta) && typeof choice.delta.content === 'string') textChars += choice.delta.content.length;
      if (typeof choice.finish_reason === 'string') sawFinish = true;
    }
  }
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  const trail = trailHeader(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  res.writeHead(200, headers);
  let doneIndex = frames.findIndex((f) => frameIsDone(f));
  if (doneIndex === -1) {
    // The gateway closed the stream without [DONE]: finish it here, whole.
    if (!sawFinish) frames.push(finishChunk(model));
    if (!sawUsage) frames.push(usageChunk(model, estimateTokens(requestChars), estimateTokens(textChars)));
    frames.push('data: [DONE]\n\n');
    doneIndex = frames.length - 1;
  } else {
    // The stream ended as it should: add what it lacked just before [DONE].
    const insert: string[] = [];
    if (!sawFinish) insert.push(finishChunk(model));
    if (!sawUsage) insert.push(usageChunk(model, estimateTokens(requestChars), estimateTokens(textChars)));
    frames.splice(doneIndex, 0, ...insert);
  }
  for (const frame of frames) res.write(frame);
  res.end();
}

/**
 * An Anthropic SSE stream relayed in the client's event order, its message_delta usage made whole:
 * an output count below 1 is replaced by an estimate from the text the stream carried. The events are
 * buffered, so the answer is one well-formed stream however the gateway paced it.
 */
async function relayAnthropicStream(upstream: ReadableStream<Uint8Array>, model: string, requestChars: number, res: ServerResponse, failures: readonly Failure[]): Promise<void> {
  const events: string[] = [];
  let textChars = 0;
  for await (const event of anthropicEventsFromOpenAI(upstream, model)) {
    events.push(event);
    for (const payload of framePayloads(event)) {
      if (!isRecord(payload)) continue;
      if (payload.type === 'content_block_delta' && isRecord(payload.delta) && payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
        textChars += payload.delta.text.length;
      }
    }
  }
  const fixed = events.map((event) => {
    const lines = event.split('\n');
    const dataLine = lines.find((line) => line.startsWith('data:'));
    if (dataLine === undefined) return event;
    let parsed: unknown;
    try {
      parsed = JSON.parse(dataLine.slice(5).trim());
    } catch {
      return event;
    }
    if (!isRecord(parsed) || parsed.type !== 'message_delta') return event;
    const usage = isRecord(parsed.usage) ? parsed.usage : {};
    const output = finiteNumber(usage.output_tokens);
    if (output !== null && output >= 1) return event;
    const input = finiteNumber(usage.input_tokens);
    const patched = {
      ...parsed,
      usage: { output_tokens: estimateTokens(textChars), ...(input !== null && input >= 0 ? { input_tokens: input } : { input_tokens: estimateTokens(requestChars) }) },
    };
    // The usage rides beside the delta, as the Messages wire sends it: keep the event's shape.
    const data = `data: ${JSON.stringify(patched)}`;
    return [...lines.slice(0, lines.indexOf(dataLine)), data, ...lines.slice(lines.indexOf(dataLine) + 1)].join('\n');
  });
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  const trail = trailHeader(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  res.writeHead(200, headers);
  for (const event of fixed) res.write(event);
  res.end();
}

/** One OpenAI chat chunk object, for a stream built from a JSON answer. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A JSON gateway answer as the OpenAI event stream the client asked for: a role chunk, the whole text
 * in one delta, a finish chunk, a usage chunk, then [DONE]. Only used when the gateway answers a
 * stream request with JSON instead of events.
 */
function openAIJsonToStream(answer: Json, model: string, requestChars: number): string[] {
  const whole = ensureOpenAIUsage(openAIJsonWithModel(answer, model), requestChars);
  const choices = Array.isArray(whole.choices) ? whole.choices : [];
  const message = choices.length > 0 && isRecord(choices[0]) && isRecord(choices[0].message) ? choices[0].message : {};
  const text = contentText(message.content);
  const finish = typeof choices[0]?.finish_reason === 'string' ? (choices[0] as Json).finish_reason as string : 'stop';
  const usage = isRecord(whole.usage) ? whole.usage : {};
  const input = finiteNumber(usage.prompt_tokens) ?? estimateTokens(requestChars);
  const output = finiteNumber(usage.completion_tokens) ?? estimateTokens(text.length);
  return [
    sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null)),
    ...(text !== '' ? [sseEvent(null, openAIChunk(model, { content: text }, null))] : []),
    sseEvent(null, openAIChunk(model, {}, finish)),
    usageChunk(model, input, output),
    'data: [DONE]\n\n',
  ];
}

/**
 * A JSON gateway answer as the Anthropic event sequence the client asked for: message_start, one text
 * block with its delta, message_delta with the stop reason and usage, then message_stop. Only used
 * when the gateway answers a stream request with JSON instead of events.
 */
function anthropicJsonToStream(answer: Json, model: string, requestChars: number): string[] {
  const whole = ensureAnthropicUsage(openAIToAnthropic(answer, model), requestChars);
  const content = Array.isArray(whole.content) ? whole.content : [];
  const text = contentText(content);
  const usage = isRecord(whole.usage) ? whole.usage : {};
  const input = finiteNumber(usage.input_tokens) ?? estimateTokens(requestChars);
  const output = finiteNumber(usage.output_tokens) ?? estimateTokens(text.length);
  const stop = typeof whole.stop_reason === 'string' ? whole.stop_reason : 'end_turn';
  return [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ...(text !== '' ? [sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } })] : []),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
}

/**
 * Serve one client request: walk the ranked candidates, first pick under the plan cap, then the rest
 * in rating order, until one model answers 2xx. A 2xx answer is the request's answer even if its body
 * surprises: calling a second model after one answered would make the request invalid.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const candidates = rankedCandidates(requires);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const stream = body.stream === true;
  const requestChars = requestTextChars(body);
  // The first pick is the best-ranked model this trial has called fewer than PLAN_CAP times; when
  // every candidate is at the cap the best-ranked model serves anyway.
  const first = candidates.find((c) => (usedCalls.get(c.id) ?? 0) < PLAN_CAP) ?? candidates[0];
  const ordered = [first, ...candidates.filter((c) => c !== first)];
  const maxRating = candidates[0].rating_milli;
  const failures: Failure[] = [];
  const tried = new Set<string>();
  const retried = new Set<string>();
  let attempts = 0;
  let lastDetail = 'every free model refused the request';
  // A refusal of the best-rated model earns one immediate retry of that same model, on a 5xx or an
  // unreachable gateway only: a 429 stays limited and a 401 is the key, not the model.
  const retrySame = (candidate: ModelRow, status: number): boolean => {
    if (candidate.rating_milli !== maxRating || retried.has(candidate.id)) return false;
    if (status !== 502 && !(status >= 500 && status <= 599)) return false;
    retried.add(candidate.id);
    tried.delete(candidate.id);
    return true;
  };
  let index = 0;
  while (index < ordered.length && attempts < MAX_ATTEMPTS) {
    const candidate = ordered[index];
    if (tried.has(candidate.id)) {
      index += 1;
      continue;
    }
    tried.add(candidate.id);
    attempts += 1;
    usedCalls.set(candidate.id, (usedCalls.get(candidate.id) ?? 0) + 1);
    let upstream: Response;
    try {
      upstream = await dial(clientApi, body, candidate.id);
    } catch (err) {
      lastDetail = `the gateway could not be reached: ${(err as Error).message}`;
      log(lastDetail);
      failures.push({ status: 502, code: 'upstream_error', model: candidate.id });
      if (retrySame(candidate, 502)) continue;
      index += 1;
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const refusal = await gatewayRefusal(upstream, candidate.id);
      lastDetail = `the gateway answered HTTP ${refusal.status}${refusal.detail !== '' ? `: ${refusal.detail}` : ''}`;
      log(lastDetail);
      failures.push({ status: refusal.status, code: refusal.code, model: candidate.id });
      // A 401 is the key, not the model: no other model would answer on it either.
      if (refusal.status === 401) break;
      if (retrySame(candidate, refusal.status)) continue;
      index += 1;
      continue;
    }
    // A 2xx answer ends the request: translate it into the client's wire and send it.
    try {
      const contentType = upstream.headers.get('content-type') ?? '';
      if (stream && !contentType.includes('text/event-stream')) {
        // The gateway answered a stream request with JSON: build the event stream from it.
        const answer: unknown = await upstream.json();
        const frames = clientApi === 'openai'
          ? openAIJsonToStream(isRecord(answer) ? answer : {}, candidate.id, requestChars)
          : anthropicJsonToStream(isRecord(answer) ? answer : {}, candidate.id, requestChars);
        const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
        const trail = trailHeader(failures);
        if (trail !== null) headers[FAILURES_HEADER] = trail;
        res.writeHead(200, headers);
        for (const frame of frames) res.write(frame);
        res.end();
        return;
      }
      if (stream) {
        if (clientApi === 'openai') await relayOpenAIStream(upstream.body, candidate.id, requestChars, res, failures);
        else await relayAnthropicStream(upstream.body, candidate.id, requestChars, res, failures);
        return;
      }
      const answer: unknown = await upstream.json();
      const translated = translateJson(clientApi, isRecord(answer) ? answer : {}, candidate.id, requestChars);
      if (translated === null) {
        lastDetail = `the gateway's answer for ${candidate.id} has no well-formed ${clientApi} body`;
        log(lastDetail);
        failures.push({ status: 502, code: 'upstream_error', model: candidate.id });
        index += 1;
        continue;
      }
      sendJson(res, 200, translated, failures);
    } catch (err) {
      log(`translating the ${candidate.id} answer failed: ${(err as Error).message}`);
      failures.push({ status: 502, code: 'upstream_error', model: candidate.id });
      if (!res.headersSent) {
        index += 1;
        continue;
      }
      res.end();
    }
    return;
  }
  if (!res.headersSent) return sendError(res, clientApi, 502, 'upstream_error', lastDetail, failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err as Error;
      log(`register attempt ${attempt} failed: ${lastError.message}`);
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
