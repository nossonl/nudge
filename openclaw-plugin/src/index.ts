// reinforceclaw openclaw plugin — thin bridge to the python backend
// catches every message in/out across all 23+ platforms, sends to localhost:8420
// all real logic (db, training, adapters) lives in python. this is just the pipe.

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const pending = new Map<string, { prompt: string; channel: string }>();

function post(host: string, path: string, body: any) {
  const secret =
    (globalThis as any).__reinforceclawSecret ??
    process.env.REINFORCECLAW_OPENCLAW_SECRET;
  fetch(`${host}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(secret ? { "X-ReinforceClaw-Secret": secret } : {}),
    },
    body: JSON.stringify(body),
  }).catch((e) => console.error("[reinforceclaw]", e.message));
}

export default definePluginEntry({
  id: "reinforceclaw-feedback",
  name: "ReinforceClaw Feedback",

  register(api) {
    const host = (api.config as any)?.reinforceclawHost ?? "http://127.0.0.1:8420";
    (globalThis as any).__reinforceclawSecret =
      (api.config as any)?.reinforceclawSecret ?? process.env.REINFORCECLAW_OPENCLAW_SECRET ?? "";

    // stash every user message — need it to pair with the bot's response
    api.on("message_received", async (event: any) => {
      const key = event.conversationId ?? event.sessionKey ?? "default";
      pending.set(key, {
        prompt: event.content,
        channel: event.channelId ?? event.metadata?.provider ?? "unknown",
      });
    });

    // bot is about to reply — pair it with the user's message, send to python
    api.on("message_sending", async (event: any, ctx: any) => {
      const key = ctx?.sessionKey ?? "default";
      const user = pending.get(key);
      if (!user || !event.content) return;

      post(host, "/feedback/capture", {
        sessionKey: key,
        prompt: user.prompt,
        response: event.content,
        channel: user.channel,
      });
      pending.delete(key);
    });

    const registerRateCommand = (name: string) => api.registerCommand({
      name,
      description: "Rate responses: /rl good, /rc good, /reinforceclaw good",
      handler: async (args: string[], ctx: any) => {
        const sub = args[0];
        if (sub === "good" || sub === "bad") {
          post(host, "/feedback/rate", {
            sessionKey: ctx.sessionKey,
            rating: sub === "good" ? 1 : -1,
          });
          return `rated: ${sub}`;
        }
        if (sub === "status") {
          try {
            const r = await fetch(`${host}/feedback/status`);
            return r.ok ? await r.text() : `error: ${r.status}`;
          } catch { return "reinforceclaw server not running"; }
        }
        return `/${name} good | /${name} bad | /${name} status`;
      },
    });
    registerRateCommand("rl");
    registerRateCommand("rc");
    registerRateCommand("reinforceclaw");
  },
});
