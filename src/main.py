from bot import Bot


def main():
    bot = Bot(
        name="QnA Bot 🤖",
        description="Ask a question and get an answer from a MongoDB Atlas collection."
    )
    bot.display_interface()


if __name__ == "__main__":
    main()