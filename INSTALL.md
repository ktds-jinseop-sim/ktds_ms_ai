# 설치 및 사용 가이드

## 1. 환경 설정

### 1.1 Python 환경
- Python 3.8 이상 필요
- 가상환경 사용 권장

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 1.2 필요한 라이브러리 설치

```bash
# 기본 라이브러리 설치
pip install gradio python-dotenv openai

# PDF 처리 라이브러리 설치
pip install docling PyPDF2 pdfplumber

# 벡터 데이터베이스 및 임베딩 라이브러리 설치
pip install faiss-cpu sentence-transformers numpy

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

## 2. Azure OpenAI 설정

### 2.1 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가:

```env
OPENAI_API_KEY=your_azure_openai_api_key
AZURE_ENDPOINT=your_azure_endpoint
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2024-02-15-preview
DEPLOYMENT_NAME=your_deployment_name
```

### 2.2 Azure OpenAI 서비스 설정
1. Azure Portal에서 OpenAI 서비스 생성
2. 모델 배포 (GPT-4 권장)
3. API 키 및 엔드포인트 확인

## 3. 실행 방법

### 3.1 기본 실행
```bash
python mvp_main.py
```

### 3.2 웹 인터페이스 접속
- 브라우저에서 `http://localhost:7860` 접속
- Gradio 인터페이스 확인

## 4. 사용 방법

### 4.1 PDF 업로드
1. "📄 PDF 업로드" 탭 선택
2. 기출문제 PDF 파일 선택
3. "PDF 업로드" 버튼 클릭
4. 처리 완료 후 "벡터 DB 통계 보기"로 확인

### 4.2 문제 생성
1. "📝 문제 풀이" 탭 선택
2. 과목 선택
3. 문제 생성 모드 선택:
   - "기출문제 기반 새 문제 생성": 기출문제를 참고한 새로운 문제
   - "기출문제 그대로 출제": 기출문제를 그대로 출제
4. "문제 생성" 버튼 클릭

### 4.3 답변 평가
1. 생성된 문제 확인
2. 답변 입력
3. "답변 확인" 버튼 클릭
4. "정답 및 해설 보기"로 상세 정보 확인

### 4.4 AI 챗봇
1. "💬 AI 챗봇" 탭 선택
2. 질문 입력
3. RAG 기반 답변 확인

## 5. 파일 구조

```
ms_ai/
├── mvp_main.py          # 메인 애플리케이션
├── prompt.py            # 프롬프트 정의
├── vector_store.py      # FAISS 벡터 스토어
├── pdf_processor.py     # PDF 처리 모듈
├── requirements.txt     # 필요한 라이브러리
├── INSTALL.md          # 설치 가이드
├── README.md           # 프로젝트 설명
├── .env                # 환경 변수 (사용자 생성)
├── faiss_vector_db/    # 벡터 데이터베이스 (자동 생성)
└── uploads/            # 업로드된 파일 (자동 생성)
```

## 6. 문제 해결

### 6.1 라이브러리 설치 오류
```bash
# FAISS 설치 오류 시
pip install faiss-cpu --no-cache-dir

# sentence-transformers 설치 오류 시
pip install sentence-transformers --no-cache-dir
```

### 6.2 메모리 부족 오류
- PDF 파일 크기 줄이기
- 청크 크기 조정 (pdf_processor.py에서 chunk_size 수정)

### 6.3 Azure OpenAI 연결 오류
- 환경 변수 확인
- API 키 및 엔드포인트 정확성 확인
- 네트워크 연결 상태 확인

## 7. 성능 최적화

### 7.1 벡터 검색 성능
- FAISS 인덱스 타입 변경 (IndexFlatL2 → IndexIVFFlat)
- 임베딩 모델 변경 (더 빠른 모델 사용)

### 7.2 PDF 처리 성능
- 청크 크기 조정
- 병렬 처리 구현

## 8. 확장 가능성

### 8.1 새로운 기능 추가
- 사용자 관리 시스템
- 학습 진도 추적
- 맞춤형 학습 경로 제안

### 8.2 다른 문서 형식 지원
- Word 문서 (.docx)
- 텍스트 파일 (.txt)
- 이미지 기반 문서 (OCR)

### 8.3 클라우드 배포
- Azure Container Apps
- Azure Functions
- Azure Kubernetes Service 