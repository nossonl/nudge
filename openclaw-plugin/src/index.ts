// nudge openclaw plugin — thin bridge to the python backend
// catches every message in/out across all 23+ platforms, sends to localhost:8420
// all real logic (db, training, adapters) lives in python. this is just the pipe.

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const pending = new Map<string, { prompt: string; channel: string }>();

function post(host: string, path: string, body: any) {
  fetch(`${host}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch((e) => console.error("[nudge]", e.message));
}

export default definePluginEntry({
  id: "nudge-feedback",
  name: "Nudge Feedback",

  register(api) {
    const host = (api.config as any)?.nudgeHost ?? "http://127.0.0.1:8420";

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

    // /rl command — works from any platform
    api.registerCommand({
      name: "rl",
      description: "Rate responses: /rl good, /rl bad, /rl status",
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
          } catch { return "nudge server not running"; }
        }
        return "/rl good | /rl bad | /rl status";
      },
    });
  },
});
