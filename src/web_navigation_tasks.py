def get_task_research_person(research_person):
    return f"open browser, search for {research_person}, navigate to their google scholar page and sort the publications by year. Then, go to their homepage if available and summarize their research interests."


def get_task_flight(origin, destination):
    return f"open browser, navigate to google flights, search for flights from {origin} to {destination}, and check the prices. departure date can be any date in the next 30days."


def get_task_restaurant(city):
    return f"open browser, search for vegetarian restaurants in {city}, and list the top-rated ones."


def get_task_person(person):
    return f"open browser, search for {person}, navigate to their Wikipedia page."


def get_task_weather(city):
    return f"open browser, search for weather in {city}, and check the humidity and temperature for the next day"


def get_task_order_food(pizza_place):
    return f"I want to know the menu of {pizza_place}. open browser, go to {pizza_place} website and check the menu. If you are asked for a location, use 07310 and choose the first location."


def get_research_persons():
    return [
        "graham neubig",
        "yoshua bengio",
        "yann lecun",
        "geoffrey hinton",
        "andrew ng",
    ]


def get_persons():
    return ["elon musk", "jack ma", "ramanujan"]


def get_origins():
    return [
        "new york",
        "san francisco",
        "los angeles",
    ]


def get_destinations():
    return [
        "paris",
        "singapore",
        "hyderabad",
    ]


def get_cities():
    return ["new york", "san francisco", "los angeles", "tokyo"]


def get_pizza_places():
    return [
        "pizzahut",
        "dominos",
        "joes pizza",
        "sbarro",
    ]


def get_all_tasks():
    persons = get_persons()
    origins = get_origins()
    destinations = get_destinations()
    cities = get_cities()
    pizza_places = get_pizza_places()
    research_persons = get_research_persons()

    tasks = []
    for person in persons:
        tasks.append(get_task_person(person))
    for origin, destination in zip(origins, destinations):
        tasks.append(get_task_flight(origin, destination))
    for city in cities:
        tasks.append(get_task_restaurant(city))
        tasks.append(get_task_weather(city))
    for pizza_place in pizza_places:
        tasks.append(get_task_order_food(pizza_place))
    for research_person in research_persons:
        tasks.append(get_task_research_person(research_person))

    return tasks
