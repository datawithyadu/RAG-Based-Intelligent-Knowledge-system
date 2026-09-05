from langchain_openai import ChatOpenAI
from langchain_classic.chains import (create_history_aware_retriever,create_retrieval_chain)
from langchain_classic.memory import ConversationBufferWindowMemory
from app.config import Config


class llm_service:
    '''Whenever an LLMService object is created, set it up with an LLM client.'''
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(
            temperature=0.7, 
            model= "gpt-3.5-turbo",
            api_key= Config.OPENAI_API_KEY
        )
        self.memory = ConversationBufferWindowMemory(
            k=3, 
            return_messages = True,
            memory_key = "chat_history"
            )
        
        self.chain = create_retrieval_chain(
            llm =self.llm,
            retriever = create_history_aware_retriever(
                vector_store = vector_store,
                memory = self.memory
            )
            )
        
    def get_response(self,query):
        try:
            response = self.chain.run(input= query)
            return response["answer"]
        except Exception as e:
            print(f"Error occurred while fetching LLM response: {e}")
            return "Sorry, I encountered an error while processing your request."