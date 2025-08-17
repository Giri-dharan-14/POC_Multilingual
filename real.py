from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai

load_dotenv()


class MultilingualAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful multilingual voice AI assistant.

CRITICAL INSTRUCTIONS:
1. AUTOMATICALLY DETECT the user's language from their speech
2. ALWAYS respond in the SAME language the user spoke to you
3. Keep ALL responses to maximum 20 words (very concise and brief)
4. Be helpful but extremely brief
5. If user switches languages, immediately switch to match their new language

Examples:
- User speaks English → Respond in English (max 20 words)
- User speaks Spanish → Respond in Spanish (max 20 words)
- User speaks French → Respond in French (max 20 words)
- User speaks Hindi → Respond in Hindi (max 20 words)
- User speaks any language → Match that language exactly

Remember: Maximum 20 words, match user's language exactly."""
        )


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral",
            temperature=0.7,
        )
    )

    await session.start(
        room=ctx.room,
        agent=MultilingualAssistant(),
        room_input_options=RoomInputOptions(
            # Remove noise cancellation for local server
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user in English initially and ask how you can help. Keep it under 20 words. After this, detect and match their language automatically."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))