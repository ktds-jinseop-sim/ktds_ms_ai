"""
설정 관리 모듈
환경 변수와 기본값을 관리합니다.
"""

import os
from typing import Union
from dotenv import load_dotenv

# .env 파일 로드 (절대 경로 사용)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

class Config:
    """설정 관리 클래스"""
    
    # Azure OpenAI 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
    OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "azure")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "")
    
    # AI 챗봇 설정
    AI_CHATBOT_TOP_K = int(os.getenv("AI_CHATBOT_TOP_K", "10"))
    AI_CHATBOT_SIMILARITY_THRESHOLD = float(os.getenv("AI_CHATBOT_SIMILARITY_THRESHOLD", "0.3"))
    AI_CHATBOT_DEBUG_LOGS = os.getenv("AI_CHATBOT_DEBUG_LOGS", "False").lower() == "true"
    AI_CHATBOT_TEMPERATURE = float(os.getenv("AI_CHATBOT_TEMPERATURE", "0.7"))
    AI_CHATBOT_MAX_TOKENS = int(os.getenv("AI_CHATBOT_MAX_TOKENS", "1500"))
    AI_CHATBOT_TOP_P = float(os.getenv("AI_CHATBOT_TOP_P", "0.9"))
    AI_CHATBOT_FREQUENCY_PENALTY = float(os.getenv("AI_CHATBOT_FREQUENCY_PENALTY", "0.0"))
    AI_CHATBOT_PRESENCE_PENALTY = float(os.getenv("AI_CHATBOT_PRESENCE_PENALTY", "0.0"))
    
    # 정보 검증 에이전트 설정
    USE_VALIDATION_AGENT = os.getenv("USE_VALIDATION_AGENT", "False").lower() == "true"
    
    # 서버 및 외부 접속 설정
    PORT = int(os.getenv("PORT", "7860"))
    USE_NGROK = os.getenv("USE_NGROK", "true").lower() == "true"
    
    # 로깅 설정
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "True").lower() == "true"
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "False").lower() == "true"
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "app.log")
    
    @classmethod
    def get_ai_chatbot_config(cls) -> dict:
        """AI 챗봇 설정 반환"""
        return {
            "top_k": cls.AI_CHATBOT_TOP_K,
            "similarity_threshold": cls.AI_CHATBOT_SIMILARITY_THRESHOLD,
            "debug_logs": cls.AI_CHATBOT_DEBUG_LOGS,
            "temperature": cls.AI_CHATBOT_TEMPERATURE,
            "max_tokens": cls.AI_CHATBOT_MAX_TOKENS,
            "top_p": cls.AI_CHATBOT_TOP_P,
            "frequency_penalty": cls.AI_CHATBOT_FREQUENCY_PENALTY,
            "presence_penalty": cls.AI_CHATBOT_PRESENCE_PENALTY
        }
    
    @classmethod
    def get_server_config(cls) -> dict:
        """서버 및 외부 접속 설정 반환"""
        return {
            "port": cls.PORT,
            "use_ngrok": cls.USE_NGROK
        }
    
    @classmethod
    def get_logging_config(cls) -> dict:
        """로깅 설정 반환"""
        return {
            "level": cls.LOG_LEVEL,
            "console": cls.LOG_TO_CONSOLE,
            "file": cls.LOG_TO_FILE,
            "file_path": cls.LOG_FILE_PATH
        } 