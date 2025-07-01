"""
정보시스템감리사 시험 문제 생성 및 평가를 위한 프롬프트 정의
"""

class ExamPrompts:
    """시험 관련 프롬프트 클래스"""
    
    # 시스템 프롬프트들
    SYSTEM_PROMPTS = {
        "question_generator": "당신은 정보시스템감리사 시험 전문가입니다. 실제 기출문제와 유사한 수준의 문제를 생성해주세요.",
        "answer_evaluator": "당신은 정보시스템감리사 시험 채점 전문가입니다. 공정하고 정확한 평가를 해주세요.",
        "chat_assistant": "당신은 정보시스템감리사 시험 준비를 도와주는 친근한 학습 도우미입니다.",
        "rag_assistant": "당신은 기출문제 데이터베이스를 활용하여 정확한 답변을 제공하는 전문가입니다."
    }
    
    @staticmethod
    def get_question_generation_prompt(subject: str, difficulty: str, question_type: str, exam_name: str = "정보시스템감리사") -> str:
        """문제 생성 프롬프트"""
        return f"""
        {exam_name} 시험에 맞는 문제를 생성해주세요.
        
        시험: {exam_name}
        과목: {subject}
        난이도: {difficulty}
        문제 유형: {question_type}
        
        다음 형식으로 응답해주세요:
        === 문제 정보 ===
        과목: {subject}
        난이도: {difficulty}
        유형: {question_type}
        
        === 문제 ===
        [문제 내용]
        
        === 보기 ===
        1) [보기1]
        2) [보기2]
        3) [보기3]
        4) [보기4]
        
        === 정답 ===
        [정답 번호]
        
        === 해설 ===
        [문제 해설 및 관련 개념 설명]
        
        === 출처 ===
        [관련 기출문제 출처 - 추후 RAG로 대체 예정]
        """
    
    @staticmethod
    def get_rag_question_generation_prompt(subject: str, difficulty: str, question_type: str, 
                                          context: str, exam_name: str = "정보시스템감리사") -> str:
        """RAG 기반 문제 생성 프롬프트"""
        return f"""
        다음 기출문제 컨텍스트를 바탕으로 {exam_name} 시험 문제를 생성해주세요.
        
        시험: {exam_name}
        과목: {subject}
        난이도: {difficulty}
        문제 유형: {question_type}
        
        === 기출문제 컨텍스트 ===
        {context}
        
        위 컨텍스트를 참고하여 다음 형식으로 응답해주세요:
        === 문제 정보 ===
        과목: {subject}
        난이도: {difficulty}
        유형: {question_type}
        출처: 기출문제 기반
        
        === 문제 ===
        [컨텍스트를 바탕으로 한 문제 내용]
        
        === 보기 ===
        1) [보기1]
        2) [보기2]
        3) [보기3]
        4) [보기4]
        
        === 정답 ===
        [정답 번호]
        
        === 해설 ===
        [문제 해설 및 관련 개념 설명 - 컨텍스트 내용 포함]
        
        === 출처 ===
        [참고한 기출문제 출처 정보]
        """
    
    @staticmethod
    def get_exact_question_prompt(context: str, exam_name: str = "정보시스템감리사") -> str:
        """기출문제 그대로 출제 프롬프트"""
        return f"""
        다음 기출문제를 그대로 출제해주세요.
        
        === 기출문제 컨텍스트 ===
        {context}
        
        위 기출문제를 다음 형식으로 정리해주세요:
        === 문제 정보 ===
        과목: [과목명]
        난이도: [난이도]
        유형: [문제 유형]
        출처: 기출문제
        
        === 문제 ===
        [기출문제 내용]
        
        === 보기 ===
        1) [보기1]
        2) [보기2]
        3) [보기3]
        4) [보기4]
        
        === 정답 ===
        [정답 번호]
        
        === 해설 ===
        [기출문제 해설]
        
        === 출처 ===
        [기출문제 출처 정보]
        """
    
    @staticmethod
    def get_answer_evaluation_prompt(question: str, user_answer: str) -> str:
        """답변 평가 프롬프트"""
        return f"""
        정보시스템감리사 시험 문제에 대한 답변을 평가해주세요:
        
        문제:
        {question}
        
        사용자 답변: {user_answer}
        
        다음 형식으로 응답해주세요:
        === 평가 결과 ===
        정답 여부: [맞음/틀림]
        점수: [점수/만점]
        
        === 피드백 ===
        [구체적인 피드백 및 개선점]
        
        === 관련 개념 ===
        [문제와 관련된 핵심 개념 설명]
        
        === 출처 ===
        [관련 기출문제 출처 - 추후 RAG로 대체 예정]
        """
    
    @staticmethod
    def get_rag_answer_evaluation_prompt(question: str, user_answer: str, context: str) -> str:
        """RAG 기반 답변 평가 프롬프트"""
        return f"""
        다음 기출문제 컨텍스트를 바탕으로 답변을 평가해주세요:
        
        === 기출문제 컨텍스트 ===
        {context}
        
        문제:
        {question}
        
        사용자 답변: {user_answer}
        
        다음 형식으로 응답해주세요:
        === 평가 결과 ===
        정답 여부: [맞음/틀림]
        점수: [점수/만점]
        
        === 피드백 ===
        [구체적인 피드백 및 개선점]
        
        === 관련 개념 ===
        [문제와 관련된 핵심 개념 설명 - 컨텍스트 내용 포함]
        
        === 출처 ===
        [참고한 기출문제 출처 정보]
        """
    
    @staticmethod
    def get_rag_question_prompt(question: str, context: str) -> str:
        """RAG 기반 질문 프롬프트"""
        return f"""
        다음 컨텍스트를 바탕으로 질문에 답변해주세요:
        
        컨텍스트:
        {context}
        
        질문: {question}
        
        답변:
        """
    
    @staticmethod
    def get_question_improvement_prompt(original_question: str, feedback: str) -> str:
        """문제 개선 프롬프트"""
        return f"""
        다음 문제를 개선해주세요:
        
        원본 문제:
        {original_question}
        
        개선 요청사항:
        {feedback}
        
        개선된 문제:
        """

class ChatPrompts:
    """챗봇 관련 프롬프트 클래스"""
    
    @staticmethod
    def get_conversation_prompt(message: str, history: list = None) -> str:
        """대화 프롬프트"""
        context = ""
        if history:
            context = "\n".join([f"사용자: {h[0]}\n도우미: {h[1]}" for h in history[-5:]])  # 최근 5개 대화만 포함
        
        return f"""
        {context}
        
        사용자: {message}
        도우미: """
    
    @staticmethod
    def get_rag_conversation_prompt(message: str, context: str, history: list = None) -> str:
        """RAG 기반 대화 프롬프트"""
        conversation_context = ""
        if history:
            conversation_context = "\n".join([f"사용자: {h[0]}\n도우미: {h[1]}" for h in history[-5:]])
        
        return f"""
        다음 기출문제 컨텍스트를 바탕으로 답변해주세요:
        
        === 기출문제 컨텍스트 ===
        {context}
        
        === 대화 기록 ===
        {conversation_context}
        
        사용자: {message}
        도우미: """

class AnalysisPrompts:
    """분석 관련 프롬프트 클래스"""
    
    @staticmethod
    def get_performance_analysis_prompt(user_answers: list) -> str:
        """성과 분석 프롬프트"""
        return f"""
        다음 사용자의 답변 기록을 분석하여 학습 성과를 평가해주세요:
        
        답변 기록:
        {user_answers}
        
        분석 결과:
        1. 전체 정답률: [%]
        2. 강점 과목: [과목명]
        3. 약점 과목: [과목명]
        4. 개선 권장사항: [구체적인 학습 방향]
        """
    
    @staticmethod
    def get_recommendation_prompt(user_profile: dict) -> str:
        """개인화 추천 프롬프트"""
        return f"""
        다음 사용자 프로필을 바탕으로 맞춤형 학습 추천을 해주세요:
        
        사용자 프로필:
        {user_profile}
        
        추천 사항:
        1. 추천 과목: [과목명]
        2. 추천 난이도: [난이도]
        3. 학습 전략: [구체적인 학습 방법]
        4. 예상 소요 시간: [시간]
        """
    
    @staticmethod
    def get_rag_recommendation_prompt(user_profile: dict, context: str) -> str:
        """RAG 기반 개인화 추천 프롬프트"""
        return f"""
        다음 기출문제 컨텍스트와 사용자 프로필을 바탕으로 맞춤형 학습 추천을 해주세요:
        
        === 기출문제 컨텍스트 ===
        {context}
        
        사용자 프로필:
        {user_profile}
        
        추천 사항:
        1. 추천 과목: [과목명]
        2. 추천 난이도: [난이도]
        3. 학습 전략: [구체적인 학습 방법 - 컨텍스트 내용 포함]
        4. 예상 소요 시간: [시간]
        5. 관련 기출문제: [컨텍스트에서 추천할 문제]
        """

class PDFProcessingPrompts:
    """PDF 처리 관련 프롬프트 클래스"""
    
    @staticmethod
    def get_pdf_summary_prompt(content: str) -> str:
        """PDF 내용 요약 프롬프트"""
        return f"""
        다음 PDF 내용을 요약해주세요:
        
        내용:
        {content}
        
        요약:
        1. 주요 주제: [주제]
        2. 핵심 개념: [개념들]
        3. 중요 내용: [중요한 내용들]
        4. 시험 관련성: [시험과의 연관성]
        """
    
    @staticmethod
    def get_pdf_question_extraction_prompt(content: str) -> str:
        """PDF에서 문제 추출 프롬프트"""
        return f"""
        다음 PDF 내용에서 시험 문제를 추출해주세요:
        
        내용:
        {content}
        
        추출된 문제들:
        [문제 형식으로 정리]
        """ 