"""IRC hook. No buttons — users type !good or !bad after a bot reply."""
# IRC can't do inline reactions so we just watch for text commands
# call on_bot_reply when your bot sends something, on_user_message on every incoming line
# source="irc"

from nudge import db


class FeedbackTracker:
    def __init__(self, model="unknown"):
        self.model = model
        # last (prompt, response) per channel — gets overwritten on next reply
        self._pending: dict[str, dict] = {}

    def on_bot_reply(self, channel: str, prompt: str, response: str):
        # store the pair; wait for !good or !bad
        self._pending[channel] = {"prompt": prompt, "response": response}

    def on_user_message(self, channel: str, text: str):
        cmd = text.strip().lower()
        if cmd not in ("!good", "!bad"):
            return
        pair = self._pending.pop(channel, None)
        if not pair:
            return  # no pending reply to rate
        rating = 1 if cmd == "!good" else -1
        conn = db.connect()
        db.add_feedback(
            conn,
            self.model,
            pair["prompt"],
            pair["response"],
            rating,
            source="irc",
        )
        conn.close()
