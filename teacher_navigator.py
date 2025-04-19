from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import logging

from task_list import (
    get_all_tasks,
    get_task_research_person,
    get_task_flight,
    get_task_restaurant,
    get_task_person,
    get_task_weather,
    get_task_order_food,
)

load_dotenv()
# Configure logging
# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get the logger for the 'browser_use' package and set it to DEBUG
browser_use_logger = logging.getLogger("browser_use")
browser_use_logger.setLevel(logging.DEBUG)


def masking_function(text):
    text.replace(os.getenv("USERNAME"), "********")
    text.replace(os.getenv("PASSWORD"), "********")
    return text


langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    mask=masking_function,
)

langfuse_handler = CallbackHandler()
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")


async def main():
    # location = input("Enter the location to search for restaurants: ")
    tasks = get_all_tasks()
    accuracy = 0
    failure = 0
    for task in tasks:
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model="gpt-4.1", callbacks=[langfuse_handler]),
            use_vision=False,
        )
        agent_history_result = await agent.run()
        if agent_history_result.is_successful():
            accuracy += 1
        else:
            failure += 1
    print(f"Accuracy: {accuracy / len(tasks)}")
    print(f"Failure: {failure / len(tasks)}")


asyncio.run(main())
