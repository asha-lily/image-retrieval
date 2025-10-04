from dotenv import dotenv_values
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate


config = dotenv_values(".env")


def main():
    model = OllamaLLM(model = config["CHAT_MODEL"])

    prompt_template = """

    You are a pizza connoisseur.

    Here are some reviews of pizza restaurants: {reviews}

    Here is a question you need to answer: {question}

    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model

    while True:
        question = input("Ask a question (q to quit): \n")
        if question == "q":
            break

        result = chain.invoke(
            {
                "reviews": [], 
                "question": question
            }
        )
        print(result)


if __name__ == "__main__":
    main()