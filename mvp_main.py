import gradio as gr
import os
from dotenv import load_dotenv
import openai
from typing import List, Dict, Any
import json
import random
import tempfile
from pathlib import Path

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# 커스텀 모듈 import
from prompt import ExamPrompts, ChatPrompts, AnalysisPrompts, PDFProcessingPrompts
from vector_store import vector_store
from pdf_processor import pdf_processor

class ExamQuestionGenerator:
    def __init__(self):
        self.conversation_history = []
        self.exam_name = "정보시스템감리사"
        self.difficulties = ["쉬움", "보통", "어려움"]
        self.question_types = ["객관식", "주관식"]
        self.current_question = None
        self.current_answer = None
        self.current_explanation = None
        self.current_context = None
        self.question_mode = "generate"  # "generate" 또는 "exact"
        
    def upload_pdf(self, pdf_file) -> str:
        """PDF 파일 업로드 및 벡터 DB 구축"""
        if pdf_file is None:
            return "❌ PDF 파일을 선택해주세요."
        
        try:
            # 임시 파일로 저장
            temp_path = tempfile.mktemp(suffix='.pdf')
            with open(temp_path, 'wb') as f:
                f.write(pdf_file.read())
            
            print(f"📄 [PDF 업로드] 파일: {pdf_file.name}")
            
            # PDF 처리
            result = pdf_processor.process_pdf(temp_path, "정보시스템감리사")
            
            # 임시 파일 삭제
            os.unlink(temp_path)
            
            if result["success"]:
                return f"✅ PDF 업로드 완료!\n\n📊 처리 결과:\n- 파일명: {result['filename']}\n- 생성된 청크: {result['chunks_count']}개\n- 과목: {result['subject']}\n\n이제 기출문제 기반 문제 생성이 가능합니다."
            else:
                return f"❌ PDF 처리 실패: {result['error']}"
                
        except Exception as e:
            error_msg = f"PDF 업로드 중 오류 발생: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def get_vector_db_stats(self) -> str:
        """벡터 DB 통계 정보"""
        try:
            stats = vector_store.get_collection_stats()
            pdf_stats = pdf_processor.get_statistics()
            
            result = "📊 벡터 DB 통계\n\n"
            result += f"📚 총 문서 수: {stats['total_documents']}\n"
            result += f"📝 시험 문제: {stats['exam_questions']}개\n"
            result += f"📖 학습 자료: {stats['study_materials']}개\n"
            result += f"❓ 사용자 질문: {stats['user_questions']}개\n"
            result += f"📋 과목: {', '.join(stats['subjects'])}\n\n"
            
            result += "📄 PDF 처리 통계\n"
            result += f"📊 총 청크: {pdf_stats['total_chunks']}\n"
            result += f"📁 PDF 소스: {', '.join(pdf_stats['pdf_sources'])}\n"
            
            return result
            
        except Exception as e:
            return f"❌ 통계 조회 실패: {e}"
    
    def generate_question(self, subject: str, question_mode: str = "generate") -> str:
        """정보시스템감리사 시험 문제 생성"""
        print(f"\n🔍 [콘솔 로그] 문제 생성 요청 - 과목: {subject}, 모드: {question_mode}")
        
        if not DEPLOYMENT_NAME:
            error_msg = "Error: DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        # 랜덤하게 난이도와 문제 유형 선택
        difficulty = random.choice(self.difficulties)
        question_type = random.choice(self.question_types)
        
        print(f"📊 [콘솔 로그] 선택된 난이도: {difficulty}, 문제 유형: {question_type}")
        
        # RAG 기반 문제 생성
        if question_mode == "generate":
            # 유사한 기출문제 검색
            search_query = f"{subject} {difficulty} {question_type}"
            similar_questions = vector_store.search_similar_questions(search_query, subject, 3)
            
            if similar_questions:
                # 컨텍스트 구성
                context = "\n\n".join([q["content"] for q in similar_questions])
                self.current_context = context
                
                prompt = ExamPrompts.get_rag_question_generation_prompt(
                    subject, difficulty, question_type, context, self.exam_name
                )
                print("🔄 [콘솔 로그] RAG 기반 문제 생성 중...")
            else:
                # RAG 결과가 없으면 일반 생성
                prompt = ExamPrompts.get_question_generation_prompt(
                    subject, difficulty, question_type, self.exam_name
                )
                self.current_context = None
                print("🔄 [콘솔 로그] 일반 문제 생성 중...")
        
        elif question_mode == "exact":
            # 기출문제 그대로 출제
            search_query = f"{subject} 기출문제"
            exact_questions = vector_store.search_similar_questions(search_query, subject, 1)
            
            if exact_questions:
                context = exact_questions[0]["content"]
                self.current_context = context
                
                prompt = ExamPrompts.get_exact_question_prompt(context, self.exam_name)
                print("🔄 [콘솔 로그] 기출문제 그대로 출제 중...")
            else:
                return "❌ 해당 과목의 기출문제를 찾을 수 없습니다. PDF를 먼저 업로드해주세요."
        
        try:
            print("🤖 [콘솔 로그] Azure OpenAI API 호출 중...")
            response = openai.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": ExamPrompts.SYSTEM_PROMPTS["question_generator"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            result = response.choices[0].message.content
            if result:
                # 결과를 파싱하여 저장
                self._parse_question_result(result)
                question_only = self._get_question_only(result)
                print("✅ [콘솔 로그] 문제 생성 완료")
                print(f"📝 [콘솔 로그] 생성된 문제:\n{question_only}")
                return question_only
            else:
                error_msg = "문제 생성에 실패했습니다."
                print(f"❌ [콘솔 로그] {error_msg}")
                return error_msg
        except Exception as e:
            error_msg = f"문제 생성 중 오류가 발생했습니다: {e}"
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
    
    def _parse_question_result(self, result: str):
        """문제 결과를 파싱하여 저장"""
        self.current_question = result
        
        # 정답과 해설 추출
        lines = result.split('\n')
        answer_section = False
        explanation_section = False
        
        for line in lines:
            if "=== 정답 ===" in line:
                answer_section = True
                explanation_section = False
                continue
            elif "=== 해설 ===" in line:
                answer_section = False
                explanation_section = True
                continue
            elif "===" in line:
                answer_section = False
                explanation_section = False
                continue
            
            if answer_section and line.strip():
                self.current_answer = line.strip()
            elif explanation_section and line.strip():
                if not self.current_explanation:
                    self.current_explanation = line.strip()
                else:
                    self.current_explanation += "\n" + line.strip()
        
        print(f"🔍 [콘솔 로그] 정답 파싱 완료: {self.current_answer}")
    
    def _get_question_only(self, result: str) -> str:
        """문제와 보기만 반환"""
        lines = result.split('\n')
        question_lines = []
        include_line = True
        
        for line in lines:
            if "=== 정답 ===" in line:
                include_line = False
                break
            if include_line:
                question_lines.append(line)
        
        return '\n'.join(question_lines)
    
    def evaluate_answer(self, user_answer: str) -> str:
        """사용자 답변 평가"""
        print(f"\n💭 [콘솔 로그] 답변 평가 요청 - 사용자 답변: '{user_answer}'")
        
        if not self.current_question:
            error_msg = "먼저 문제를 생성해주세요."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        if not self.current_answer:
            error_msg = "정답 정보를 찾을 수 없습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        # RAG 기반 평가
        if self.current_context:
            prompt = ExamPrompts.get_rag_answer_evaluation_prompt(
                self.current_question, user_answer, self.current_context
            )
        else:
            prompt = ExamPrompts.get_answer_evaluation_prompt(
                self.current_question, user_answer
            )
        
        try:
            print("🤖 [콘솔 로그] Azure OpenAI API 호출 중...")
            response = openai.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": ExamPrompts.SYSTEM_PROMPTS["answer_evaluator"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            result = response.choices[0].message.content
            if result:
                print(f"✅ [콘솔 로그] 답변 평가 완료")
                return result
            else:
                error_msg = "답변 평가에 실패했습니다."
                print(f"❌ [콘솔 로그] {error_msg}")
                return error_msg
        except Exception as e:
            error_msg = f"답변 평가 중 오류가 발생했습니다: {e}"
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
    
    def show_solution(self) -> str:
        """정답 및 해설 표시"""
        print("\n🔍 [콘솔 로그] 정답 및 해설 요청")
        
        if not self.current_question:
            error_msg = "먼저 문제를 생성해주세요."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        if not self.current_answer or not self.current_explanation:
            error_msg = "정답 및 해설 정보를 찾을 수 없습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            return error_msg
        
        solution = f"""
=== 정답 ===
{self.current_answer}

=== 해설 ===
{self.current_explanation}

=== 출처 ===
{self.current_context if self.current_context else "[관련 기출문제 출처 - 추후 RAG로 대체 예정]"}
        """
        
        print(f"📖 [콘솔 로그] 정답 및 해설 표시:\n{solution.strip()}")
        return solution.strip()
    
    def chat_with_ai(self, message: str, history: List[List[str]]) -> tuple[List[List[str]], str]:
        """AI와의 일반적인 대화"""
        print(f"\n💬 [콘솔 로그] AI 챗봇 메시지: '{message}'")
        
        if not DEPLOYMENT_NAME:
            error_msg = "Error: DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다."
            print(f"❌ [콘솔 로그] {error_msg}")
            history.append([message, error_msg])
            return history, ""
        
        # RAG 기반 대화 시도
        try:
            # 관련 컨텍스트 검색
            similar_chunks = vector_store.search_similar_questions(message, n_results=2)
            context = ""
            if similar_chunks:
                context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
            
            if context:
                prompt = ChatPrompts.get_rag_conversation_prompt(message, context, history)
                print("🔄 [콘솔 로그] RAG 기반 대화 중...")
            else:
                prompt = ChatPrompts.get_conversation_prompt(message, history)
                print("🔄 [콘솔 로그] 일반 대화 중...")
            
            print("🤖 [콘솔 로그] Azure OpenAI API 호출 중...")
            response = openai.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": ExamPrompts.SYSTEM_PROMPTS["chat_assistant"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            ai_response = response.choices[0].message.content
            if ai_response:
                print(f"🤖 [콘솔 로그] AI 응답: {ai_response}")
                history.append([message, ai_response])
            else:
                error_msg = "응답을 생성할 수 없습니다."
                print(f"❌ [콘솔 로그] {error_msg}")
                history.append([message, error_msg])
            return history, ""
        except Exception as e:
            error_msg = f"오류가 발생했습니다: {e}"
            print(f"❌ [콘솔 로그] {error_msg}")
            history.append([message, error_msg])
            return history, ""

# 인스턴스 생성
generator = ExamQuestionGenerator()

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="정보시스템감리사 문제 생성 챗봇") as demo:
        gr.Markdown("# 🎯 정보시스템감리사 시험 문제 생성 및 질의 응답 챗봇")
        gr.Markdown("Azure OpenAI와 LangGraph를 활용한 맞춤형 학습 시스템")
        
        with gr.Tabs():
            # 탭 1: PDF 업로드
            with gr.TabItem("📄 PDF 업로드"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### PDF 업로드")
                        gr.Markdown("기출문제 PDF를 업로드하면 벡터 DB에 저장되어 RAG 기반 문제 생성이 가능합니다.")
                        
                        pdf_upload = gr.File(
                            label="기출문제 PDF 업로드",
                            file_types=[".pdf"],
                            type="binary"
                        )
                        upload_btn = gr.Button("PDF 업로드", variant="primary")
                        
                    with gr.Column():
                        upload_output = gr.Textbox(
                            label="업로드 결과",
                            lines=8,
                            interactive=False
                        )
                
                with gr.Row():
                    stats_btn = gr.Button("벡터 DB 통계 보기", variant="secondary")
                    stats_output = gr.Textbox(
                        label="벡터 DB 통계",
                        lines=6,
                        interactive=False
                    )
                
                # 이벤트 연결
                upload_btn.click(
                    fn=generator.upload_pdf,
                    inputs=[pdf_upload],
                    outputs=upload_output
                )
                
                stats_btn.click(
                    fn=generator.get_vector_db_stats,
                    inputs=[],
                    outputs=stats_output
                )
            
            # 탭 2: 문제 생성 및 답변
            with gr.TabItem("📝 문제 풀이"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 시험 정보")
                        gr.Markdown(f"**시험명**: {generator.exam_name}")
                        gr.Markdown("**난이도**: 랜덤 선택 (쉬움/보통/어려움)")
                        gr.Markdown("**문제 유형**: 랜덤 선택 (객관식/주관식)")
                        
                        subject_input = gr.Dropdown(
                            choices=[
                                "정보시스템 일반",
                                "정보시스템 감리",
                                "정보시스템 보안",
                                "정보시스템 운영",
                                "정보시스템 개발",
                                "정보시스템 구축",
                                "정보시스템 관리",
                                "정보시스템 설계"
                            ],
                            label="과목 선택",
                            value="정보시스템 일반"
                        )
                        
                        question_mode = gr.Radio(
                            choices=[
                                ("기출문제 기반 새 문제 생성", "generate"),
                                ("기출문제 그대로 출제", "exact")
                            ],
                            label="문제 생성 모드",
                            value="generate"
                        )
                        
                        generate_btn = gr.Button("문제 생성", variant="primary")
                    
                    with gr.Column():
                        question_output = gr.Textbox(
                            label="문제",
                            lines=12,
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column():
                        user_answer_input = gr.Textbox(
                            label="답변 입력",
                            placeholder="답변을 입력하세요...",
                            lines=2
                        )
                        evaluate_btn = gr.Button("답변 확인", variant="secondary")
                    
                    with gr.Column():
                        evaluation_output = gr.Textbox(
                            label="결과",
                            lines=6,
                            interactive=False
                        )
                
                with gr.Row():
                    solution_btn = gr.Button("정답 및 해설 보기", variant="secondary")
                    solution_output = gr.Textbox(
                        label="정답 및 해설",
                        lines=10,
                        interactive=False
                    )
                
                # 이벤트 연결
                generate_btn.click(
                    fn=generator.generate_question,
                    inputs=[subject_input, question_mode],
                    outputs=question_output
                )
                
                evaluate_btn.click(
                    fn=generator.evaluate_answer,
                    inputs=[user_answer_input],
                    outputs=evaluation_output
                )
                
                solution_btn.click(
                    fn=generator.show_solution,
                    inputs=[],
                    outputs=solution_output
                )
            
            # 탭 3: AI 챗봇
            with gr.TabItem("💬 AI 챗봇"):
                chatbot = gr.Chatbot(
                    label="정보시스템감리사 학습 도우미와 대화하기",
                    height=400
                )
                msg = gr.Textbox(
                    label="메시지",
                    placeholder="정보시스템감리사 관련 질문이나 도움이 필요한 내용을 입력하세요...",
                    lines=2
                )
                clear_btn = gr.Button("대화 초기화")
                
                def user_input(message, history):
                    return "", history + [[message, None]]
                
                def bot_response(history):
                    if history[-1][1] is None:
                        history[-1][1] = generator.chat_with_ai(history[-1][0], history[:-1])[0][-1][1]
                    return history
                
                msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot_response, chatbot, chatbot
                )
                clear_btn.click(lambda: None, None, chatbot, queue=False)
        
        # 하단 정보
        gr.Markdown("---")
        gr.Markdown("### 🔧 기술 스택")
        gr.Markdown("- **Azure OpenAI**: GPT-4 기반 자연어 처리")
        gr.Markdown("- **Gradio**: 사용자 인터페이스")
        gr.Markdown("- **Docling**: PDF 텍스트 추출")
        gr.Markdown("- **FAISS**: 벡터 데이터베이스")
        gr.Markdown("- **RAG**: 기출문제 검색 및 질의 응답")
        gr.Markdown("- **LangGraph**: 멀티 에이전트 시스템 (추후 구현)")
        
    return demo

if __name__ == "__main__":
    print("🎯 [콘솔 로그] 정보시스템감리사 문제 생성 챗봇 시작")
    print("🌐 [콘솔 로그] Gradio 웹 인터페이스 실행 중...")
    demo = create_gradio_interface()
    demo.launch(share=False, debug=True, show_error=True)