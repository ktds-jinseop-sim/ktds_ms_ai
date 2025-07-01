# kt ds MS AI 역량향상과정 MVP

### 1. 주제 선정
##### RAG 기반 시험 문제 생성 및 질의 응답 챗봇
Azure 클라우드 서비스와 LangGraph 기반의 멀티 에이전트 시스템을 활용하여, 기존 기출문제 데이터를 기반으로 새로운 문제를 생성하고, 사용자 질문에 대해 정답 여부와 출처 정보를 제공하는 챗봇 시스템 구축.

### 2. 개요 및 목적
  - 개요
    기존 기출문제 데이터를 바탕으로 RAG를 활용하여 새로운 시험 문제를 생성하고, 사용자로부터 입력된 답변에 대해 정답 여부를 판단하며, 관련 기출문제의 출처를 알려주는 AI 챗봇을 개발.
  - 목적
    - 학습자에게 맞춤형 시험 문제 제공
    - 자동화된 피드백 및 학습 경로 제안
    - 문제 출처 및 해설 제공으로 신뢰도 향상
  - 활용 Azure 서비스
    - Azure OpenAI: GPT-4 기반 RAG 및 자연어 처리
    - Azure Cognitive Search: 문서 인덱싱 및 검색
    - Azure Functions: 이벤트 기반 서버리스 백엔드 로직
    - Azure Blob Storage: 기출문제 데이터 저장
    - Azure Cosmos DB 또는 Table Storage: 문제와 사용자 기록 저장
    - Azure Container Apps or AKS: 앱 배포
    - Azure Monitor & App Insights: 모니터링

### 3. 아키텍처
[User (Gradio UI)]<br>
        ↓<br>
[PDF 업로드] → [Docling PDF 처리] → [FAISS 벡터 DB]<br>
        ↓<br>
[Azure Functions API]<br>
        ↓<br>
[LangGraph Agent]<br>
        ↓<br>
[Prompt Engineering / Pythonic Prompt]<br>
        ↓<br>
[Azure OpenAI (RAG)]  <--- [FAISS 벡터 검색 (기출문제)]<br>
        ↓<br>
[Answer & 출처 정보]<br>
        ↓<br>
[Gradio UI 표시]<br>

### 4. 구현 시 고려사항
#### 기능 구현
- PDF 업로드 및 벡터 DB 구축 (Docling + FAISS)
- 기출문제 그대로 출제 또는 기반 문제 생성
- 사용자 입력 정답 판별
- 출처 제공 (기출문제 연결)
- 사용자 UI 통한 상호작용 (Gradio)
#### 기술 구현
- 모든 백엔드 로직은 Azure Functions / LangGraph / Azure OpenAI로 구성
- 코드 구조는 모듈화 (PDF 처리, 문제 생성, 답안 평가, 출처 추적 모듈 분리)
- FAISS 벡터 검색 및 ranking 최적화
#### UX
- Gradio 인터페이스로 쉽고 직관적인 경험 제공
- 주요 흐름: PDF 업로드 → 벡터 DB 구축 → 문제 생성 → 답변 제출 → 정답/출처 반환
- 틀린 오답 문제 개인 저장, 리마인드, 누적
- 카테고리별 문제 추적 및 DB화
#### 혁신성 및 확장성
- 독창성
  - Prompt 모듈화로 재사용성 강화 (Pythonic Prompting)
  - LangGraph 기반 멀티 에이전트 구조 활용
  - Docling을 활용한 효율적인 PDF 처리
- 확장성
  - 시험별 다양한 조건 기반 문제 생성 확장 가능
  - 사용자 분석 기반 맞춤형 학습 경로 제안 기능 추가 가능
  - 다양한 문서 형식 지원 확장 가능

---
### 평가 기준 base
#### 1. 기능 구현
- RAG 사용
- PDF 업로드 및 벡터 DB 구축
#### 2. 기술 구현
- Azure 기반
- 코드의 모듈화
- FAISS 벡터 DB 활용
#### 3. UX
- Gradio 사용(핵심 기능 실행 및 절차 간편)
#### 4. 발표 및 데모
- 문제점
- 솔루션
- 아키텍처
- 질의 응답
#### 5. 혁신성 및 확장성
- 독창적인 부분 -> pythonic prompt 사용
- Langgraph 사용 agent 기반
- Docling을 활용한 PDF 처리