import os
import json

API_KEY = ""
with open("./API_KEY.txt", "r") as f:
    API_KEY = f.read()
os.environ["OPENAI_API_KEY"] = API_KEY

from llama_index import (
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context,
    StorageContext,
    load_index_from_storage,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.memory import ChatMemoryBuffer
from llama_index.prompts import (
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=512)
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)
memory = ChatMemoryBuffer.from_defaults(token_limit=2500)

# TODO: Figure out how to wrap chat in template
template = (
    "Below is the information about the courses relavent to the student's query. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer in a way that the student can gain a deeper understanding about their query: {query_str}\n"
)

documents = SimpleDirectoryReader("./data").load_data()
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)


# index = VectorStoreIndex(nodes, show_progress=True)
# index.storage_context.persist(persist_dir="./persist")

storage_context = StorageContext.from_defaults(persist_dir="./backend/persist")
index = load_index_from_storage(storage_context)

query_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="""Your task is to assist in generating compelling and informative property listings for both rental and sales properties. You'll be provided with various details about each property, such as location, number of bedrooms and bathrooms, amenities, and terms for rental properties or additional features for sale properties."
Guidelines
Highlight Marketable Features: Focus on the elements that make the property attractive to potential renters or buyers. This includes unique amenities, the number of bedrooms and bathrooms, and any special terms or policies.
Be Clear and Concise: Make sure the listing is easy to read and understand, avoiding unnecessary jargon.
Balance Detail and Brevity: Provide enough detail to interest potential renters or buyers, but keep it brief enough to ensure the listing is easily scannable.
Adapt Style: If the property is luxurious, make the language upscale. If it's a cozy, family home, make the tone warm and inviting.
Differentiate Between Rental and Sales: For rental properties, emphasize terms and pet policies. For properties for sale, provide information on the interior and exterior features, as well as school and tax details.",
""",
)

query = input("Enter query or type 'quit' to exit: ")
while query.lower() != "quit":
    response = query_engine.chat(query)
    print(response)
    query = input("\nAsk a follow up or type 'quit' to exit: ")
