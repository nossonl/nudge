// reinforceclaw openclaw plugin: forwards local OpenClaw feedback to the ReinforceClaw bridge.

import { Buffer } from "node:buffer";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

type PendingPrompt = { prompt: string; channel: string; context: unknown; userId: string; ts: number };
type CapturedTurn = PendingPrompt & { response: string; sessionKey: string; pendingKey?: string };
const pending = new Map<string, PendingPrompt[]>();
const failedCaptures = new Map<string, CapturedTurn>();
const capturesInFlight = new Set<string>();
const capturedMessages = new Map<string, CapturedTurn>();
const capturedMessageIds = new Map<string, CapturedTurn>();
const capturedPollIds = new Map<string, CapturedTurn>();
const ambiguousMessageIds = new Map<string, number>();
const capturedResponses = new Map<string, CapturedTurn>();
const capturedPendingKeys = new Map<string, CapturedTurn>();
const handledOutbound = new Map<string, number>();
const handledInbound = new Map<string, number>();
const promptedPendingKeys = new Map<string, number>();
let bridgeSecret = "";
const adminLastRun = new Map<string, number>();
const SAFE_COMMANDS = new Set(["good", "bad", "undo", "status"]);
const ADMIN_COMMANDS = new Set(["train", "rollback", "reset", "on", "off"]);
const GOOD_ALIASES = new Set([
  "good", "yes", "y", "+1", "up", "thumbs up", "thumbsup", "thumbs_up", "check", "white check mark",
  "white_check_mark", "heavy check mark", "heavy_check_mark", "✅", "✔", "👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿",
]);
const BAD_ALIASES = new Set([
  "bad", "no", "n", "-1", "down", "thumbs down", "thumbsdown", "thumbs_down", "x", "cross", "no entry",
  "no_entry", "negative squared cross mark", "negative_squared_cross_mark", "❌", "✖", "👎", "👎🏻", "👎🏼", "👎🏽", "👎🏾", "👎🏿",
]);
const MAX_PENDING = 1000;
const MAX_PENDING_AGE_MS = 30 * 60 * 1000;
const DEDUPE_AGE_MS = 2 * 60 * 1000;
const MAX_TEXT_BYTES = 2_000_000;
const MAX_CONTEXT_MESSAGE_BYTES = 120_000;
const MAX_TRACKED_TURNS = 2_000;
const BRIDGE_TIMEOUT_MS = 15_000;
const TRUNCATION_MARKER = "\n...[reinforceclaw truncated oversized text]\n";
const DEFAULT_HOST = "http://127.0.0.1:8420";
const PERMANENT_CAPTURE_REJECTS = new Set([400, 411, 413, 414, 415]);

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function idOf(value: any, depth = 0): string {
  if (depth > 4 || value === undefined || value === null) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  if (typeof value === "object") return idOf(value.messageId ?? value.id ?? value.name ?? value.key ?? value.threadId ?? value.channelId ?? value.conversationId, depth + 1);
  return String(value);
}

function sessionKey(event: any, ctx?: any) {
  return idOf(ctx?.sessionKey) || idOf(event?.sessionKey) || idOf(event?.conversationId) ||
    idOf(event?.threadId) || idOf(event?.target) || channelOf(event) || "default";
}

function outboundMessageIdOf(...values: any[]) {
  for (const value of values) {
    const id = idOf(
      value?.messageId ?? value?.messageID ?? value?.providerMessageId ?? value?.clientMessageId ??
      value?.id ?? value?.key ?? value?.targetMessageId ?? value?.targetId ?? value?.ts ?? value?.timestamp ??
      value?.message ?? value?.targetMessage ?? value?.item ?? value?.callback,
    );
    if (id) return id;
  }
  return "";
}

function reactionMessageIdOf(event: any, ctx?: any) {
  return outboundMessageIdOf(
    event?.targetMessage ?? event?.message ?? event?.item ?? event?.callback,
    {
      messageId: event?.targetMessageId ?? event?.targetId ?? event?.reaction?.messageId ??
        event?.reaction?.targetMessageId ?? ctx?.messageId ?? ctx?.callback?.messageId,
    },
  );
}

function outboundMessageKey(event: any, ctx?: any) {
  const id = outboundMessageIdOf(event, ctx?.message, ctx?.callback);
  return id ? `${sessionKey(event, ctx)}#${id}` : "";
}

function reactionMessageKey(event: any, ctx?: any) {
  const id = reactionMessageIdOf(event, ctx);
  return id ? `${sessionKey(event, ctx)}#${id}` : "";
}

function contentOf(value: any, depth = 0): string {
  if (depth > 4 || value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map((item) => contentOf(item, depth + 1)).filter(Boolean).join("\n");
  if (typeof value !== "object") return String(value);
  for (const key of ["content", "text", "body", "caption", "message"]) {
    const text = contentOf(value[key], depth + 1);
    if (text) return text;
  }
  return "";
}

function capText(text: string, maxBytes = MAX_TEXT_BYTES) {
  if (Buffer.byteLength(text, "utf8") <= maxBytes) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const keep = Math.ceil((lo + hi) / 2);
    const head = Math.floor(keep / 2);
    const candidate = `${text.slice(0, head)}${TRUNCATION_MARKER}${text.slice(-(keep - head))}`;
    if (Buffer.byteLength(candidate, "utf8") <= maxBytes) lo = keep;
    else hi = keep - 1;
  }
  if (lo <= 0) return TRUNCATION_MARKER;
  const head = Math.floor(lo / 2);
  return `${text.slice(0, head)}${TRUNCATION_MARKER}${text.slice(-(lo - head))}`;
}

function cleanText(value: unknown) {
  return String(value || "")
    .replace(/\ufe0f/g, "")
    .trim()
    .toLowerCase()
    .replace(/^[`*_.!:\s]+|[`*_.!:\s]+$/g, "")
    .replace(/\s+/g, " ");
}

function normalizeCommand(value: unknown, allowBare = true) {
  const text = cleanText(value);
  for (const prefix of ["/rl ", "/rc ", "/reinforceclaw "]) {
    if (text.startsWith(prefix)) return normalizeCommand(text.slice(prefix.length), true);
  }
  if (text.startsWith("/")) return normalizeCommand(text.slice(1), true);
  if (!allowBare) return "";
  if (GOOD_ALIASES.has(text)) return "good";
  if (BAD_ALIASES.has(text)) return "bad";
  return text;
}

function isReinforceCommandText(value: unknown) {
  const text = cleanText(value);
  return ["/rl", "/rc", "/reinforceclaw"].includes(text) ||
    ["/rl ", "/rc ", "/reinforceclaw "].some((prefix) => text.startsWith(prefix));
}

function reactionOf(event: any) {
  const reaction = event?.reaction;
  if (typeof reaction === "string" || typeof reaction === "number") return String(reaction);
  return String(
    event?.emoji ?? event?.reactionName ?? reaction?.emoji ?? reaction?.name ??
    reaction?.text ?? event?.name ?? contentOf(event),
  );
}

function pollIdOf(value: any) {
  return idOf(
    value?.pollId ?? value?.poll_id ?? value?.poll?.id ?? value?.poll?.pollId ??
    value?.pollAnswer?.poll_id ?? value?.pollAnswer?.pollId ?? value?.poll_answer?.poll_id ??
    value?.answer?.pollId ?? value?.answer?.poll_id,
  );
}

function pollChoiceOf(event: any) {
  const text = contentOf(event?.option ?? event?.choice ?? event?.vote ?? event?.pollVote ?? event?.selectedOption ?? event?.selected_option).trim();
  if (text) return normalizeCommand(text);
  const texts = event?.optionTexts ?? event?.options ?? event?.selectedOptions ?? event?.selected_options;
  if (Array.isArray(texts) && texts.length) {
    const chosen = contentOf(texts[0]).trim();
    if (chosen) return normalizeCommand(chosen);
  }
  const ids = event?.optionIds ?? event?.option_ids ?? event?.pollAnswer?.option_ids ?? event?.poll_answer?.option_ids;
  const index = Array.isArray(ids) ? Number(ids[0]) : Number(event?.optionId ?? event?.option_id ?? event?.optionIndex ?? event?.option_index);
  return index === 0 ? "good" : index === 1 ? "bad" : "";
}

function stopDelivery(event: any, ctx: any) {
  for (const item of [event, ctx]) {
    try { item?.preventDefault?.(); } catch {}
    try { item?.stopPropagation?.(); } catch {}
  }
}

function lowerField(obj: any, names: string[]) {
  for (const name of names) {
    const value = obj?.[name] ?? obj?.metadata?.[name];
    if (value !== undefined && value !== null) return String(value).toLowerCase();
  }
  return "";
}

function trueField(obj: any, names: string[]) {
  return names.some((name) => obj?.[name] === true || obj?.metadata?.[name] === true);
}

function feedbackMode(cfg: Record<string, unknown>) {
  const raw = String(cfg.reinforceclawFeedbackMode || "slash_reactions_prompt")
    .trim().toLowerCase().replace(/[-\s]+/g, "_");
  if (["slash", "slash_only", "commands", "commands_only"].includes(raw)) return "slash_only";
  if (["slash_reactions_prompt", "slash_reactions_poll", "all", "default"].includes(raw)) return "slash_reactions_prompt";
  return "slash_and_reactions";
}

function feedbackPromptsEnabled(cfg: Record<string, unknown>) {
  return feedbackMode(cfg) === "slash_reactions_prompt";
}

function roleOf(event: any, ctx?: any) {
  return lowerField(event, ["role", "senderRole", "authorRole", "messageRole", "type"]) ||
    lowerField(ctx, ["role", "senderRole", "authorRole"]);
}

function isHumanInbound(event: any, ctx?: any) {
  const role = roleOf(event, ctx);
  const direction = lowerField(event, ["direction", "messageDirection"]);
  if (["assistant", "bot", "system", "tool", "model"].includes(role)) return false;
  if (["out", "outbound", "sent", "sending"].includes(direction)) return false;
  if (trueField(event, ["bot", "isBot", "fromBot", "fromSelf", "outgoing", "system"]) ||
      trueField(ctx, ["bot", "isBot", "fromBot", "fromSelf", "outgoing", "system"])) return false;
  return true;
}

function isAssistantOutbound(event: any, ctx?: any) {
  const role = roleOf(event, ctx);
  const direction = lowerField(event, ["direction", "messageDirection"]);
  if (["user", "human", "system", "tool"].includes(role)) return false;
  if (["in", "inbound", "received"].includes(direction)) return false;
  if (role) return ["assistant", "bot", "model"].includes(role);
  return trueField(event, ["fromSelf", "outgoing", "isAssistant", "isBot"]) ||
    trueField(ctx, ["fromSelf", "outgoing", "isAssistant", "isBot"]);
}

function channelOf(event: any) {
  return idOf(event?.channelId) || idOf(event?.channel) || idOf(event?.metadata?.provider) || "openclaw";
}

function providerOf(event: any, ctx?: any) {
  const raw = [
    event?.metadata?.provider, event?.provider, event?.channel, event?.messageChannel,
    ctx?.messageChannel, ctx?.channel, ctx?.provider,
  ].map((x) => String(x || "").toLowerCase()).join(" ");
  for (const provider of ["telegram", "whatsapp", "discord", "slack"]) {
    if (raw.includes(provider)) return provider;
  }
  return "";
}

function outboundTarget(event: any, ctx?: any) {
  return idOf(event?.to) || idOf(event?.target) || idOf(event?.conversationId) || idOf(ctx?.conversationId) ||
    idOf(event?.channelId) || idOf(ctx?.channelId) || idOf(ctx?.threadId);
}

function accountIdOf(event: any, ctx?: any) {
  return idOf(event?.accountId) || idOf(event?.metadata?.accountId) || idOf(ctx?.accountId);
}

function rolloutContext(event: any, ctx: any, prompt: string) {
  const raw = Array.isArray(ctx?.messages) ? ctx.messages : Array.isArray(event?.messages) ? event.messages : [];
  const messages = raw
    .map((m: any) => ({ role: String(m?.role || ""), content: capText(contentOf(m).trim(), MAX_CONTEXT_MESSAGE_BYTES) }))
    .filter((m: any) => ["user", "assistant"].includes(m.role) && m.content)
    .slice(-12);
  if (!messages.some((m: any) => m.role === "user" && m.content === prompt)) {
    messages.push({ role: "user", content: prompt });
  }
  return { messages, session: sessionKey(event, ctx), channel: channelOf(event) };
}

function commandParts(call: any[]) {
  const first = call[0] ?? {};
  const args = Array.isArray(first) ? first : Array.isArray(first.args) ? first.args : [];
  return { args, ctx: call[1] ?? first };
}

function secretHeader(): Record<string, string> {
  const secret = bridgeSecret;
  return secret ? { "X-ReinforceClaw-Secret": String(secret) } : {};
}

function pluginConfig(api: any) {
  return ((api as any).pluginConfig ?? {}) as Record<string, unknown>;
}

function secretFrom(cfg: Record<string, unknown>) {
  return String(cfg.reinforceclawSecret || "");
}

function localHost(value: unknown) {
  const raw = String(value || DEFAULT_HOST);
  try {
    const url = new URL(raw);
    if (url.protocol === "http:" && ["127.0.0.1", "localhost", "::1", "[::1]"].includes(url.hostname)) {
      return url.origin;
    }
  } catch {}
  return DEFAULT_HOST;
}

function callerId(ctx: any) {
  return idOf(ctx?.userId) || idOf(ctx?.senderId) || idOf(ctx?.authorId) || idOf(ctx?.fromId) ||
    idOf(ctx?.user) || idOf(ctx?.sender) || idOf(ctx?.author) || idOf(ctx?.from);
}

function actorId(event: any, ctx?: any) {
  return callerId(event) || callerId(ctx) || idOf(event?.reactorId) || idOf(event?.actorId) || idOf(event?.user?.id);
}

function allowedRaters(cfg: Record<string, unknown>) {
  return String(cfg.reinforceclawRaterUsers || cfg.reinforceclawAdminUsers || "")
    .split(",").map((x) => x.trim()).filter(Boolean);
}

function canRateTurn(turn: CapturedTurn, event: any, ctx: any, cfg: Record<string, unknown>) {
  const systemReaction = event?.reinforceclawReactionSystem === true;
  const blockedFields = systemReaction ? ["bot", "isBot", "fromBot", "fromSelf", "outgoing"] :
    ["bot", "isBot", "fromBot", "fromSelf", "outgoing", "system"];
  if (trueField(event, blockedFields) || trueField(ctx, blockedFields)) return false;
  const actor = actorId(event, ctx);
  if (actor && allowedRaters(cfg).includes(actor)) return true;
  if (turn.userId) return !!actor && actor === turn.userId;
  const group = trueField(event, ["isGroup", "group"]) || trueField(ctx, ["isGroup", "group"]);
  return !group;
}

function scopedKey(base: string, userId: string) {
  return userId ? `${base}::${encodeURIComponent(userId)}` : base;
}

function responseSig(response: string) {
  return `${response.length}:${response.slice(0, 256)}`;
}

function responseLookupKey(key: string, response: string) {
  return `${key}:${responseSig(response)}`;
}

function rememberCapture(key: string, turn: CapturedTurn, event?: any, ctx?: any) {
  capturedResponses.set(responseLookupKey(key, turn.response), turn);
  if (turn.pendingKey) capturedPendingKeys.set(turn.pendingKey, turn);
  const msgId = event ? outboundMessageIdOf(event, ctx?.message, ctx?.callback) : "";
  if (msgId) {
    const existing = capturedMessageIds.get(msgId);
    if (existing && existing !== turn) {
      capturedMessageIds.delete(msgId);
      ambiguousMessageIds.set(msgId, Date.now());
    } else if (!ambiguousMessageIds.has(msgId)) {
      capturedMessageIds.set(msgId, turn);
    }
  }
  const msgKey = event ? outboundMessageKey(event, ctx) : "";
  if (msgKey) capturedMessages.set(msgKey, turn);
}

function capturedForResponse(base: string, userId: string, response: string) {
  const keys = [scopedKey(base, userId), base].filter(Boolean);
  for (const key of keys) {
    const turn = capturedResponses.get(responseLookupKey(key, response));
    if (turn) return turn;
  }
  const suffix = `:${responseSig(response)}`;
  const matches = [...capturedResponses.entries()]
    .filter(([key]) => key.startsWith(`${base}::`) && key.endsWith(suffix))
    .map(([, turn]) => turn);
  return matches.length === 1 ? matches[0] : undefined;
}

function pendingFor(base: string, userId: string) {
  const key = scopedKey(base, userId);
  const direct = pending.get(key);
  if (direct?.length) return { key, user: direct[0] };
  if (userId) return undefined;
  const matches = [...pending.entries()].filter(([k, q]) => q.length && (k === base || k.startsWith(`${base}::`)));
  return matches.length === 1 ? { key: matches[0][0], user: matches[0][1][0] } : undefined;
}

function popPending(key: string) {
  const queue = pending.get(key);
  if (!queue?.length) return;
  queue.shift();
  if (!queue.length) pending.delete(key);
}

function pendingCount() {
  let n = 0;
  for (const q of pending.values()) n += q.length;
  return n;
}

function rememberDedupe(map: Map<string, number>, key: string) {
  if (key) map.set(key, Date.now());
}

function seenDedupe(map: Map<string, number>, key: string) {
  return !!key && map.has(key);
}

function outboundDedupeKey(event: any, ctx: any, base: string, response: string) {
  return outboundMessageKey(event, ctx) || (response ? `${base}#${responseSig(response)}` : "");
}

function inboundDedupeKey(event: any, ctx: any, text: string) {
  return outboundMessageIdOf(event, ctx) || `${sessionKey(event, ctx)}#${callerId(ctx) || callerId(event)}#${text}`;
}

function reactionSystemEvent(event: any) {
  const text = contentOf(event);
  const match = text.match(/\breaction\s+(?:added|add):\s*(.+?)\s+by\b.*?\bmsg\s+([^\s]+)/i);
  const contextKey = String(event?.metadata?.contextKey || event?.contextKey || "");
  if (!match && !contextKey.includes(":reaction:")) return null;
  const parts = contextKey.split(":");
  let emoji = match?.[1]?.trim() || "";
  let messageId = match?.[2]?.trim() || "";
  let userId = "";
  if (parts.length >= 5 && parts[1] === "reaction") {
    emoji = emoji || parts[parts.length - 1];
    userId = parts[parts.length - 2] || "";
    if (parts[0] === "discord") messageId = messageId || parts[3];
    else if (parts[0] === "slack") messageId = messageId || parts[4];
    else if (parts[0] === "telegram") messageId = messageId || parts[4];
    else messageId = messageId || parts[parts.length - 3];
  }
  return emoji && messageId ? { reaction: emoji, targetMessageId: messageId, userId } : null;
}

function dropOldestPending() {
  let oldestKey = "";
  let oldestTs = Infinity;
  for (const [key, q] of pending) {
    if (q[0] && q[0].ts < oldestTs) {
      oldestKey = key;
      oldestTs = q[0].ts;
    }
  }
  if (oldestKey) popPending(oldestKey);
}

function adminAllowed(sub: string, ctx: any, cfg: Record<string, unknown>) {
  if (!ADMIN_COMMANDS.has(sub) || ![true, "true", "1"].includes(cfg.reinforceclawAllowAdminCommands as any)) return false;
  const allowed = String(cfg.reinforceclawAdminUsers || "")
    .split(",").map((x) => x.trim()).filter(Boolean);
  return allowed.length > 0 && allowed.includes(callerId(ctx));
}

function adminRateLimited(sub: string, ctx: any) {
  if (!ADMIN_COMMANDS.has(sub)) return false;
  const key = `${callerId(ctx)}:${sub}`;
  const now = Date.now();
  const last = adminLastRun.get(key) || 0;
  if (now - last < 30_000) return true;
  adminLastRun.set(key, now);
  return false;
}

function prunePending() {
  const cutoff = Date.now() - MAX_PENDING_AGE_MS;
  for (const [key, queue] of pending) {
    while (queue.length && queue[0].ts < cutoff) queue.shift();
    if (!queue.length) pending.delete(key);
  }
  for (const [key, item] of failedCaptures) {
    if (item.ts < cutoff) failedCaptures.delete(key);
  }
  for (const [key, item] of capturedMessages) {
    if (item.ts < cutoff) capturedMessages.delete(key);
  }
  for (const [key, item] of capturedMessageIds) {
    if (item.ts < cutoff) capturedMessageIds.delete(key);
  }
  for (const [key, item] of capturedResponses) {
    if (item.ts < cutoff) capturedResponses.delete(key);
  }
  for (const [key, item] of capturedPendingKeys) {
    if (item.ts < cutoff) capturedPendingKeys.delete(key);
  }
  for (const [key, ts] of ambiguousMessageIds) {
    if (ts < cutoff) ambiguousMessageIds.delete(key);
  }
  const dedupeCutoff = Date.now() - DEDUPE_AGE_MS;
  for (const [key, ts] of handledOutbound) {
    if (ts < dedupeCutoff) handledOutbound.delete(key);
  }
  for (const [key, ts] of handledInbound) {
    if (ts < dedupeCutoff) handledInbound.delete(key);
  }
  for (const [key, ts] of promptedPendingKeys) {
    if (ts < cutoff) promptedPendingKeys.delete(key);
  }
  trimTrackedMaps();
}

function trimMap<T>(map: Map<string, T>, max = MAX_TRACKED_TURNS) {
  if (map.size <= max) return;
  const rows = [...map.entries()].map(([key, value]) => [
    key,
    typeof value === "number" ? value : Number((value as any)?.ts || 0),
  ] as [string, number]);
  rows.sort((a, b) => a[1] - b[1]);
  for (const [key] of rows.slice(0, map.size - max)) map.delete(key);
}

function trimTrackedMaps() {
  for (const map of [
    failedCaptures, capturedMessages, capturedMessageIds, capturedPollIds,
    ambiguousMessageIds, capturedResponses, capturedPendingKeys, handledOutbound,
    handledInbound, promptedPendingKeys,
  ] as Map<string, any>[]) trimMap(map);
}

async function bridgeReady(host: string) {
  try {
    const r = await fetchWithTimeout(`${host}/feedback/status`, { headers: secretHeader() });
    return r.ok;
  } catch {
    return false;
  }
}

async function waitForBridge(host: string) {
  for (let i = 0; i < 10; i++) {
    if (await bridgeReady(host)) return true;
    await sleep(200);
  }
  return false;
}

function fetchWithTimeout(url: string, init: RequestInit = {}, timeoutMs = BRIDGE_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...init, signal: controller.signal }).finally(() => clearTimeout(timer));
}

function post(host: string, path: string, body: any) {
  return fetchWithTimeout(`${host}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...secretHeader(),
    },
    body: JSON.stringify(body),
  });
}

async function command(host: string, body: any) {
  try {
    const r = await post(host, "/feedback/command", body);
    const text = await r.text();
    let payload: any = {};
    try { payload = text ? JSON.parse(text) : {}; } catch {}
    const message = payload.message;
    if (!r.ok) return message || text || `error: ${r.status}`;
    return typeof message === "string" ? message : JSON.stringify(message);
  } catch {
    return "reinforceclaw server not running";
  }
}

export default definePluginEntry({
  id: "reinforceclaw-feedback",
  name: "ReinforceClaw Feedback",

  register(api: any) {
    const cfg = pluginConfig(api);
    const host = localHost(cfg.reinforceclawHost);
    const secret = secretFrom(cfg);
    bridgeSecret = secret;

    const ensureBridge = () => bridgeReady(host);

    const captureTurn = async (key: string, turn: CapturedTurn) => {
      try {
        if (!(await ensureBridge())) {
          failedCaptures.set(key, turn);
          return "";
        }
        const r = await post(host, "/feedback/capture", {
          sessionKey: key,
          prompt: turn.prompt,
          response: turn.response,
          channel: turn.channel,
          context: turn.context,
        });
        const text = await r.text();
        let payload: any = {};
        try { payload = text ? JSON.parse(text) : {}; } catch {}
        if (r.ok) {
          const pendingKey = String(payload.key || "");
          failedCaptures.delete(key);
          turn.pendingKey = pendingKey || turn.pendingKey;
          rememberCapture(key, turn);
          return pendingKey;
        }
        if (PERMANENT_CAPTURE_REJECTS.has(r.status)) {
          failedCaptures.delete(key);
          return "";
        }
        failedCaptures.set(key, turn);
        console.error("[reinforceclaw]", `capture failed: ${r.status}`);
        return "";
      } catch (e: any) {
        failedCaptures.set(key, turn);
        console.error("[reinforceclaw]", e.message);
        return "";
      }
    };

    const on = (names: string[], handler: (...args: any[]) => Promise<void>) => {
      for (const name of names) {
        try { api.on(name, handler); } catch {}
      }
    };

    const rateLatest = async (key: string, sub: string, pendingKey = "") => {
      if (!(await ensureBridge())) return "reinforceclaw server not running";
      if (!pendingKey && failedCaptures.has(key)) {
        pendingKey = await captureTurn(key, failedCaptures.get(key)!) || "";
      }
      return await command(host, { sessionKey: key, command: sub, pendingKey });
    };

    const commandKey = (event: any, ctx: any) => scopedKey(sessionKey(event, ctx), callerId(ctx) || callerId(event));

    const handleCommandText = async (text: string, event: any, ctx: any) => {
      const sub = normalizeCommand(text);
      if (!(SAFE_COMMANDS.has(sub) || adminAllowed(sub, ctx, cfg))) return false;
      const dedupeKey = inboundDedupeKey(event, ctx, `cmd:${sub}`);
      if (seenDedupe(handledInbound, dedupeKey)) return true;
      rememberDedupe(handledInbound, dedupeKey);
      if (adminRateLimited(sub, ctx)) return true;
      await rateLatest(commandKey(event, ctx), sub);
      return true;
    };

    const cleanupRatedTurn = (turn: CapturedTurn, reactedKey: string, reactedId: string) => {
      capturedMessages.delete(reactedKey);
      if (reactedId) capturedMessageIds.delete(reactedId);
      capturedResponses.delete(responseLookupKey(turn.sessionKey, turn.response));
      if (turn.pendingKey) capturedPendingKeys.delete(turn.pendingKey);
    };

    const rememberPoll = (result: any, turn: CapturedTurn, event?: any, ctx?: any) => {
      const pollId = pollIdOf(result) || pollIdOf(event) || pollIdOf(ctx);
      const messageId = outboundMessageIdOf(result, event, ctx?.message, ctx?.callback);
      if (pollId) capturedPollIds.set(pollId, turn);
      if (messageId) {
        capturedMessageIds.set(messageId, turn);
        capturedMessages.set(`${turn.sessionKey}#${messageId}`, turn);
      }
    };

    const handlePollVote = async (event: any, ctx: any) => {
      const pollId = pollIdOf(event) || pollIdOf(ctx);
      const messageId = reactionMessageIdOf(event, ctx) || outboundMessageIdOf(event, ctx?.message, ctx?.callback);
      const turn = (pollId && capturedPollIds.get(pollId)) ||
        (messageId && !ambiguousMessageIds.has(messageId) ? capturedMessageIds.get(messageId) : undefined);
      if (!turn || !canRateTurn(turn, event, ctx, cfg)) return false;
      const rating = pollChoiceOf(event);
      if (rating !== "good" && rating !== "bad") return false;
      const actor = actorId(event, ctx);
      const dedupeKey = `poll:${pollId || messageId}:${actor}:${rating}`;
      if (seenDedupe(handledInbound, dedupeKey)) return true;
      rememberDedupe(handledInbound, dedupeKey);
      const pendingKey = turn.pendingKey || await captureTurn(turn.sessionKey, turn);
      if (!pendingKey) return false;
      const reply = await command(host, { sessionKey: turn.sessionKey, command: rating, pendingKey });
      if (String(reply).startsWith("rated:")) cleanupRatedTurn(turn, `${turn.sessionKey}#${messageId}`, messageId);
      return true;
    };

    const handleReactionRating = async (event: any, ctx: any) => {
      if (feedbackMode(cfg) === "slash_only") return false;
      const rating = normalizeCommand(reactionOf(event));
      if (rating !== "good" && rating !== "bad") return false;
      prunePending();
      const reactedKey = reactionMessageKey(event, ctx);
      const reactedId = reactionMessageIdOf(event, ctx);
      const turn = capturedMessages.get(reactedKey) ||
        (reactedId && !ambiguousMessageIds.has(reactedId) ? capturedMessageIds.get(reactedId) : undefined);
      if (!turn || !canRateTurn(turn, event, ctx, cfg)) return false;
      const dedupeKey = `reaction:${reactedId || reactedKey}:${actorId(event, ctx)}:${rating}`;
      if (seenDedupe(handledInbound, dedupeKey)) return true;
      rememberDedupe(handledInbound, dedupeKey);
      const pendingKey = turn.pendingKey || await captureTurn(turn.sessionKey, turn);
      if (!pendingKey) return false;
      const reply = await command(host, { sessionKey: turn.sessionKey, command: rating, pendingKey });
      if (String(reply).startsWith("rated:")) cleanupRatedTurn(turn, reactedKey, reactedId);
      return true;
    };

    const payloadFor = (rating: "good" | "bad", pendingKey: string) => `reinforceclaw:${rating}:${pendingKey}`;

    const sendFeedbackPrompt = async (turn: CapturedTurn, event: any, ctx: any) => {
      if (!feedbackPromptsEnabled(cfg) || !turn.pendingKey || promptedPendingKeys.has(turn.pendingKey)) return;
      const provider = providerOf(event, ctx);
      const target = outboundTarget(event, ctx);
      if (!provider || !target) return;
      promptedPendingKeys.set(turn.pendingKey, Date.now());
      const good = payloadFor("good", turn.pendingKey);
      const bad = payloadFor("bad", turn.pendingKey);
      const accountId = accountIdOf(event, ctx) || undefined;
      const replyTo = outboundMessageIdOf(event, ctx?.message, ctx?.callback) || undefined;
      const poll = { question: "Rate this response", options: ["Good", "Bad"], maxSelections: 1 };
      try {
        if (provider === "telegram") {
          const sendPoll = api.runtime?.channel?.telegram?.sendPollTelegram;
          if (sendPoll) {
            rememberPoll(await sendPoll(target, poll, { accountId, replyToMessageId: replyTo, isAnonymous: false }), turn, event, ctx);
            return;
          }
          const send = api.runtime?.channel?.telegram?.sendMessageTelegram;
          if (!send) return promptedPendingKeys.delete(turn.pendingKey);
          await send(target, "Rate this response:", {
            accountId,
            replyToMessageId: replyTo,
            buttons: [[
              { text: "Good", callback_data: good },
              { text: "Bad", callback_data: bad },
            ]],
          });
        } else if (provider === "discord") {
          const sendPoll = api.runtime?.channel?.discord?.sendPollDiscord;
          if (sendPoll) {
            rememberPoll(await sendPoll(target, poll, { accountId, content: "Rate this response:" }), turn, event, ctx);
            return;
          }
          const send = api.runtime?.channel?.discord?.sendComponentMessage;
          if (!send) return promptedPendingKeys.delete(turn.pendingKey);
          await send(target, {
            text: "Rate this response:",
            reusable: false,
            blocks: [{ type: "actions", buttons: [
              { label: "Good", style: "success", callbackData: good },
              { label: "Bad", style: "danger", callbackData: bad },
            ] }],
          }, { accountId, replyTo });
        } else if (provider === "slack") {
          const send = api.runtime?.channel?.slack?.sendMessageSlack;
          if (!send) return promptedPendingKeys.delete(turn.pendingKey);
          await send(target, "Rate this response:", {
            accountId,
            threadTs: replyTo,
            blocks: [{
              type: "actions",
              block_id: `reinforceclaw_${turn.pendingKey.slice(0, 24)}`,
              elements: [
                { type: "button", action_id: "openclaw:reply_button", text: { type: "plain_text", text: "Good", emoji: true }, style: "primary", value: good },
                { type: "button", action_id: "openclaw:reply_button", text: { type: "plain_text", text: "Bad", emoji: true }, style: "danger", value: bad },
              ],
            }],
          });
        } else if (provider === "whatsapp") {
          const send = api.runtime?.channel?.whatsapp?.sendPollWhatsApp;
          if (!send) return promptedPendingKeys.delete(turn.pendingKey);
          rememberPoll(await send(target, poll, { accountId, verbose: false }), turn, event, ctx);
        }
      } catch {
        promptedPendingKeys.delete(turn.pendingKey);
      }
    };

    // Stash every user message so the outbound hook can pair it with the bot response.
    on(["message_received"], async (event: any, ctx: any) => {
      const prompt = capText(contentOf(event).trim());
      if (!prompt || !isHumanInbound(event, ctx)) return;
      prunePending();
      if (isReinforceCommandText(prompt)) {
        await handleCommandText(prompt, event, ctx);
        stopDelivery(event, ctx);
        return;
      }
      const key = commandKey(event, ctx);
      while (pendingCount() >= MAX_PENDING) dropOldestPending();
      const queue = pending.get(key) || [];
      queue.push({
        prompt,
        channel: channelOf(event),
        context: rolloutContext(event, ctx, prompt),
        userId: callerId(ctx) || callerId(event),
        ts: Date.now(),
      });
      pending.set(key, queue);
    });

    // Pair each bot reply with the user's message and send it to Python.
    on(["message_sending", "message_sent"], async (event: any, ctx: any) => {
      const base = sessionKey(event, ctx);
      prunePending();
      if (!isAssistantOutbound(event, ctx)) return;
      const match = pendingFor(base, callerId(ctx) || callerId(event)) || pendingFor(base, "");
      const user = match?.user;
      const response = capText(contentOf(event).trim());
      if (!user && response) {
        const remembered = capturedForResponse(base, callerId(ctx) || callerId(event), response);
        if (remembered) {
          rememberCapture(remembered.sessionKey, remembered, event, ctx);
          await sendFeedbackPrompt(remembered, event, ctx);
        }
      }
      if (!user || !response) return;
      const dedupeKey = outboundDedupeKey(event, ctx, base, response);
      if (seenDedupe(handledOutbound, dedupeKey)) return;
      rememberDedupe(handledOutbound, dedupeKey);
      const captureKey = `${match!.key}:${response.length}:${response.slice(0, 64)}`;
      if (capturesInFlight.has(captureKey)) return;
      capturesInFlight.add(captureKey);
      popPending(match!.key);

      try {
        const turn = { ...user, response, sessionKey: match!.key, ts: Date.now() };
        await captureTurn(match!.key, turn);
        rememberCapture(match!.key, turn, event, ctx);
        await sendFeedbackPrompt(turn, event, ctx);
      } finally {
        capturesInFlight.delete(captureKey);
      }
    });

    try {
      api.on("inbound_claim", async (event: any, ctx: any) => {
        const text = capText(contentOf(event).trim());
        if (text && isReinforceCommandText(text)) {
          await handleCommandText(text, event, ctx);
          return { handled: true };
        }
        const reaction = reactionSystemEvent(event);
        if (await handlePollVote(event, ctx)) return { handled: true };
        if (reaction && await handleReactionRating({ ...event, ...reaction, reinforceclawReactionSystem: true }, ctx)) return { handled: true };
      });
    } catch {}

    const interactiveCommand = (ctx: any) => {
      const raw = String(
        ctx?.callback?.payload ?? ctx?.callback?.data ?? ctx?.interaction?.payload ?? ctx?.interaction?.data ??
        ctx?.interaction?.value ?? ctx?.interaction?.values?.[0] ?? "",
      );
      let data: any = raw;
      try { data = raw ? JSON.parse(raw) : raw; } catch {}
      if (typeof data === "string" && data.startsWith("reinforceclaw:")) {
        const parts = data.split(":");
        return { sub: normalizeCommand(parts[1]), pendingKey: parts.slice(2).join(":") };
      }
      const commandText = typeof data === "object" ? data.command ?? data.rating ?? data.value : data;
      const pendingKey = typeof data === "object" ? data.pendingKey ?? data.key : raw.split(":").slice(1).join(":");
      return { sub: normalizeCommand(String(commandText).split(":")[0]), pendingKey: String(pendingKey || "") };
    };

    const registerInteractive = (channel: "telegram" | "discord" | "slack") => {
      try {
        api.registerInteractiveHandler({
          channel,
          namespace: "reinforceclaw",
          handler: async (ictx: any) => {
            if (ictx?.auth && ictx.auth.isAuthorizedSender === false) return { handled: true };
            const { sub, pendingKey } = interactiveCommand(ictx);
            if (sub !== "good" && sub !== "bad") return { handled: false };
            const turn = pendingKey ? capturedPendingKeys.get(pendingKey) : undefined;
            if (pendingKey && (!turn || !canRateTurn(turn, { senderId: idOf(ictx.senderId) }, ictx, cfg))) return { handled: true };
            const key = turn?.sessionKey || scopedKey(idOf(ictx.conversationId) || idOf(ictx.parentConversationId) || channel, idOf(ictx.senderId));
            if (!(await ensureBridge())) return { handled: true };
            const text = await command(host, { sessionKey: key, command: sub, pendingKey });
            const reply = channel === "slack" ? { text, responseType: "ephemeral" } : { text, ephemeral: true };
            await (ictx.respond?.reply?.(reply) ?? ictx.respond?.followUp?.(reply) ?? Promise.resolve());
            return { handled: true };
          },
        });
      } catch {}
    };
    registerInteractive("telegram");
    registerInteractive("discord");
    registerInteractive("slack");

    const registerRateCommand = (name: string) => api.registerCommand({
      name,
      acceptsArgs: true,
      requireAuth: true,
      description: "Rate ReinforceClaw: good, bad, undo, status",
      handler: async (...call: any[]) => {
        const { args, ctx } = commandParts(call);
        const sub = normalizeCommand(args[0]);
        if (SAFE_COMMANDS.has(sub) || adminAllowed(sub, ctx, cfg)) {
          if (adminRateLimited(sub, ctx)) return { text: "admin command rate limited" };
          const key = scopedKey(sessionKey(ctx, ctx), callerId(ctx));
          if (!(await ensureBridge())) return { text: "reinforceclaw server not running" };
          let pendingKey = "";
          if (["good", "bad"].includes(sub) && failedCaptures.has(key)) {
            pendingKey = await captureTurn(key, failedCaptures.get(key)!) || "";
          }
          return { text: await command(host, {
            sessionKey: key,
            command: sub,
            pendingKey,
          }) };
        }
        return { text: `/${name} good | bad | undo | status` };
      },
    });
    registerRateCommand("rl");
    registerRateCommand("rc");
    registerRateCommand("reinforceclaw");
  },
});
