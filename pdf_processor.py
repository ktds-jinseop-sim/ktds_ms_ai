"""
PDF 처리 및 벡터화 모듈
Docling을 사용한 PDF 텍스트 추출 및 FAISS 벡터 DB 구축
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime
import json

# Docling 관련 import
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Docling이 설치되지 않았습니다. pip install docling을 실행해주세요.")
    DocumentConverter = None

# FAISS 관련 import
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("FAISS 또는 sentence-transformers가 설치되지 않았습니다.")
    print("pip install faiss-cpu sentence-transformers를 실행해주세요.")
    faiss = None
    SentenceTransformer = None

class PDFProcessor:
    """PDF 처리 및 벡터화 클래스"""
    
    def __init__(self, vector_db_path: str = "faiss_vector_db"):
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # 벡터 모델 초기화
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """벡터 모델 및 FAISS 인덱스 초기화"""
        if SentenceTransformer is None or faiss is None:
            print("필요한 라이브러리가 설치되지 않았습니다.")
            return
        
        try:
            # 한국어에 특화된 임베딩 모델 사용
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # FAISS 인덱스 초기화 (L2 거리 기반)
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dimension)
            
            print(f"✅ 벡터 모델 초기화 완료 (차원: {dimension})")
        except Exception as e:
            print(f"❌ 벡터 모델 초기화 실패: {e}")
    
    def process_pdf(self, pdf_file_path: str, subject: str = "정보시스템감리사") -> Dict[str, Any]:
        """PDF 파일 처리 및 벡터화"""
        print(f"\n📄 [PDF 처리] 파일: {pdf_file_path}")
        
        if DocumentConverter is None:
            return {"success": False, "error": "Docling 라이브러리가 설치되지 않았습니다."}
        
        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_file_path)
            # 전체 텍스트 추출 (Markdown 기준)
            full_text = result.document.export_to_markdown()
            # 텍스트 청크 분할 및 벡터화는 기존 로직 재사용
            text_chunks = self._extract_and_chunk_text_from_text(full_text, subject)
            if not text_chunks:
                return {"success": False, "error": "PDF에서 텍스트를 추출할 수 없습니다."}
            self._vectorize_and_store(text_chunks, subject, pdf_file_path)
            self._save_metadata()
            print(f"✅ PDF 처리 완료 - {len(text_chunks)}개 청크 생성")
            return {
                "success": True,
                "chunks_count": len(text_chunks),
                "subject": subject,
                "filename": Path(pdf_file_path).name
            }
        except Exception as e:
            error_msg = f"PDF 처리 중 오류 발생: {e}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _extract_and_chunk_text_from_text(self, full_text: str, subject: str) -> List[Dict[str, Any]]:
        """텍스트(문자열)에서 청크 분할"""
        chunks = []
        try:
            chunk_size = 500
            overlap = 100
            for i in range(0, len(full_text), chunk_size - overlap):
                chunk_text = full_text[i:i + chunk_size]
                if len(chunk_text.strip()) < 50:
                    continue
                chunk_id = hashlib.md5(f"{subject}_{i}_{chunk_text[:50]}".encode()).hexdigest()
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text.strip(),
                    "start_pos": i,
                    "end_pos": i + len(chunk_text),
                    "subject": subject,
                    "created_at": datetime.now().isoformat()
                })
        except Exception as e:
            print(f"텍스트 추출 중 오류: {e}")
        return chunks
    
    def _vectorize_and_store(self, chunks: List[Dict[str, Any]], subject: str, pdf_file_path: str):
        """청크를 벡터화하고 FAISS에 저장"""
        if self.embedding_model is None or self.index is None:
            print("벡터 모델이 초기화되지 않았습니다.")
            return
        
        try:
            # 텍스트 추출
            texts = [chunk["text"] for chunk in chunks]
            
            # 벡터화
            print("🔄 텍스트 벡터화 중...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # FAISS 인덱스에 추가
            self.index.add(embeddings.astype('float32'))
            
            # 메타데이터 저장
            for i, chunk in enumerate(chunks):
                chunk["embedding_id"] = len(self.documents) + i
                chunk["pdf_source"] = Path(pdf_file_path).name
                self.documents.append(chunk["text"])
                self.metadata.append(chunk)
            
            print(f"✅ {len(chunks)}개 청크 벡터화 완료")
            
        except Exception as e:
            print(f"벡터화 중 오류: {e}")
    
    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """유사한 청크 검색"""
        if self.embedding_model is None or self.index is None:
            return []
        
        try:
            # 쿼리 벡터화
            query_embedding = self.embedding_model.encode([query])
            
            # FAISS 검색
            distances, indices = self.index.search(query_embedding.astype('float32'), n_results)
            
            # 결과 포맷팅
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    result = {
                        "rank": i + 1,
                        "distance": float(distance),
                        "text": self.documents[idx],
                        "metadata": self.metadata[idx]
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"검색 중 오류: {e}")
            return []
    
    def get_chunks_by_subject(self, subject: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """과목별 청크 조회"""
        results = []
        
        for metadata in self.metadata:
            if metadata.get("subject") == subject:
                results.append({
                    "text": metadata.get("text", ""),
                    "metadata": metadata
                })
                if len(results) >= n_results:
                    break
        
        return results
    
    def _save_metadata(self):
        """메타데이터를 파일로 저장"""
        try:
            metadata_file = self.vector_db_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_chunks": len(self.documents),
                    "metadata": self.metadata,
                    "last_updated": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            
            # FAISS 인덱스 저장
            index_file = self.vector_db_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))
            
            print(f"✅ 메타데이터 및 인덱스 저장 완료")
            
        except Exception as e:
            print(f"메타데이터 저장 중 오류: {e}")
    
    def load_existing_data(self):
        """기존 데이터 로드"""
        try:
            metadata_file = self.vector_db_path / "metadata.json"
            index_file = self.vector_db_path / "faiss_index.bin"
            
            if metadata_file.exists() and index_file.exists():
                # 메타데이터 로드
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", [])
                    self.documents = [meta.get("text", "") for meta in self.metadata]
                
                # FAISS 인덱스 로드
                self.index = faiss.read_index(str(index_file))
                
                print(f"✅ 기존 데이터 로드 완료 - {len(self.documents)}개 청크")
                return True
            
        except Exception as e:
            print(f"기존 데이터 로드 중 오류: {e}")
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """벡터 DB 통계 정보"""
        stats = {
            "total_chunks": len(self.documents),
            "total_metadata": len(self.metadata),
            "index_size": self.index.ntotal if self.index else 0,
            "subjects": list(set([meta.get("subject", "") for meta in self.metadata if meta.get("subject")])),
            "pdf_sources": list(set([meta.get("pdf_source", "") for meta in self.metadata if meta.get("pdf_source")]))
        }
        return stats
    
    def clear_all_data(self):
        """모든 데이터 삭제"""
        try:
            self.documents = []
            self.metadata = []
            if self.index:
                self.index.reset()
            
            # 파일 삭제
            metadata_file = self.vector_db_path / "metadata.json"
            index_file = self.vector_db_path / "faiss_index.bin"
            
            if metadata_file.exists():
                metadata_file.unlink()
            if index_file.exists():
                index_file.unlink()
            
            print("✅ 모든 데이터 삭제 완료")
            
        except Exception as e:
            print(f"데이터 삭제 중 오류: {e}")

# 전역 PDF 프로세서 인스턴스
pdf_processor = PDFProcessor() 