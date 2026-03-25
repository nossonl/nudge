"""MS Teams hook. Wraps a bot, appends Good/Bad card after every reply."""
# uses botbuilder-python (Microsoft Bot Framework SDK)
# card click → nudge db. source="teams" so you can filter it out.

from nudge import db

try:
    from botbuilder.core import ActivityHandler, TurnContext
    from botbuilder.schema import Activity, ActivityTypes
except ImportError:
    ActivityHandler = object
    TurnContext = object
    Activity = ActivityTypes = None

# adaptive card skeleton — two buttons, one action each
_CARD = {
    "type": "AdaptiveCard",
    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
    "version": "1.4",
    "body": [{"type": "TextBlock", "text": "Was that helpful?", "weight": "Bolder"}],
    "actions": [
        {"type": "Action.Submit", "title": "Good", "data": {"nudge_rating": 1}},
        {"type": "Action.Submit", "title": "Bad",  "data": {"nudge_rating": -1}},
    ],
}


def _rating_card():
    if Activity is None or ActivityTypes is None:
        raise ImportError("botbuilder-core and botbuilder-schema are required for nudge.hooks.teams")
    return Activity(
        type=ActivityTypes.message,
        attachments=[{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": _CARD,
        }],
    )


class NudgeActivityHandler(ActivityHandler):
    # wrap your real bot handler by subclassing this instead of ActivityHandler
    def __init__(self, model="unknown"):
        if Activity is None:
            raise ImportError("botbuilder-core and botbuilder-schema are required for nudge.hooks.teams")
        super().__init__()
        self.model = model
        self._pending = {}  # conversation_id → last user text

    async def on_message_activity(self, turn_context: TurnContext):
        user_text = turn_context.activity.text or ""
        data = (turn_context.activity.value or {})

        # incoming click from the card
        if "nudge_rating" in data:
            conv_id = turn_context.activity.conversation.id
            pair = self._pending.pop(conv_id, None)
            if pair and pair[0] != "__waiting__":
                conn = db.connect()
                db.add_feedback(conn, self.model, pair[0], pair[1],
                                data["nudge_rating"], source="teams")
                conn.close()
            await turn_context.send_activity("Got it, thanks.")
            return

        # normal message — let subclass handle it, then attach the card
        conv_id = turn_context.activity.conversation.id
        self._pending[conv_id] = ("__waiting__", "__waiting__")

        await self.handle_message(turn_context, user_text)

    async def handle_message(self, turn_context: TurnContext, text: str):
        raise NotImplementedError("override handle_message")

    async def _capture_reply(self, turn_context: TurnContext,
                             prompt: str, response: str):
        # records pair, sends reply + rating card
        conv_id = turn_context.activity.conversation.id
        self._pending[conv_id] = (prompt, response)
        await turn_context.send_activity(response)
        await turn_context.send_activity(_rating_card())
