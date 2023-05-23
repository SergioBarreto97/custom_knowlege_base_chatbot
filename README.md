# Custom Knowledge Base ChatBot
This project helps connect custom knowledge bases to the OpenAI gpt-3.5-turbo api. A user can then ask a question and gpt-3.5-turbo can provide a response based on the custom knowledge base. This is useful for using gpt-3.5-turbo with subjects that gpt-3.5-turbo would otherwise be unfamiliar with.

## Problem Overview
If gpt-3.5-turbo api had no limit on the size of queries and if bandwidth costs could be ignored, connecting a custom knowledge base to gpt-3.5-turbo would be really easy. For any user question, you would simply send gpt-3.5-turbo the question and include your entire custom knowledge base as context. However, this isn't the reality we live it. gpt-3.5-turbo has a limit of 4096 total tokens (1 token is roughly 1 word) between the query and response. Even if this wasn't the case, you still wouldn't want to send your entire custom knowledge base in all queries, as this would be bandwidth expensive and would increase latency. So, we need to find a way to send gpt-3.5-turbo only the relavent portions of our custom knowledge base when a user makes a question.

## Solution Overview
In order to only send relavent portions of our custom knowledge base when a user makes a question, we first must split up our custom knowledge base into smaller snippets that can be passed along to gpt-3.5-turbo. We then need to identify the relavent snippets to pass along to gpt-3.5-turbo whenever a user makes a question. In order to identify the relavent snippets, we must first convert all our snippets into vector embeddings (using OpenAI's text-embedding-ada-002 api). These snippet vector embeddings are then stored in a Pinecone database. The user question itself must also then be converted into a vector embedding. We can then query our Pinecone database to see which of our snippet vector embeddings are the most similar to the question vector embedding. This can be done using a few different formulas but cosine similarity will work best for our use case (text-embedding-ada-002 embeddings). You can read more about similarity search [here](https://www.pinecone.io/learn/what-is-similarity-search/). By the way, there are ways to conduct a similarity search with strings but these searches are much faster and effective using embeddings which is why we have taken this approach.

## Example Project: WallStreetOdds.com Support Chatbot
This project is based on the WallStreetOdds.com Support Chatbot that I integrated and can be viewed on the website. For reference, WallStreetOdds.com is a platform that provides advanced alternative trading data and tools for investors. On the website, the chatbot is integrated using JavaScript fetch calls to an Azure Function that hosts the answer_user_question.py logic.

## Customizable
This project is intended to be easily customizable to meet your own needs.

## Requirements
**Python Libraries**: Please see requirements.txt for necessary python libraries.
**Additional Requirements**: You will need to open an OpenAI and Pinecone account to get started. You will also need to save your OpenAI api key in your environment variables under the name OPENAI_KEY. You will likewise, need to add your Pinecone api key in your environment variables under the name PINECONE_KEY.

## Usage
**support_snippets folder**: Add your custom knowledge base support snippets here.
**add_embeddings.py**: Follow along the script altering necessary variables/logic and run the file to convert all your support snippets into vector embeddings. This is necessary in order to be able to run answer_user_question.py.
**answer_user_question.py**: Follow along the script altering necessary variables/logic and run the script's get_question_response(question) function to ask gpt-3.5-turbo questions regarding your custom knowlege base.

## Questions & Suggestions
For any questions or inquiries, please contact sergiobarreto9797@gmail.com.
