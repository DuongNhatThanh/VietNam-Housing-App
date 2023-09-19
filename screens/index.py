from screens.search import Search_Property
from screens.chat_bot import run_chatbot
from screens.chat_bot_2 import run_chatbot_2
from utils.index import get_hash

def get_routes():
    screens = [
        
        {
            "component": Search_Property,
            "name": "Search",
            "icon": "search"  
        },
        {
            "component": run_chatbot,
            "name": "Chatbot (news,law)",
            "icon": "chat"  
        },
        {
            "component": run_chatbot_2,
            "name": "Chatbot (property)",
            "icon": "chat"  
        }
    ]
    
    return get_hash(screens)
