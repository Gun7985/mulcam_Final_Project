# 식중독 질문 
# 열이 나는 것도 식중독의 증상 중 하나일까? // custom
# 식중독이 있을 경우 먹어야 하는 약 알려줘 (발표 커스텀)
# 탈수 증상이 있는데 뭘 마셔도 될까?
# 식중독일 때 어떤 종류의 병원을 가야해? //custom
# 해운대에서 회를 먹었는데 약간 상태가 맛이 갔나 계속 토를 하고 화장실도 계속 가 이게 뭐때문에 그런거야??
# 관절염 
# 관절염의 주요 증상은 무엇인가요?
# 급성 관절염과 만성 관절염의 차이점은 무엇인가요?
# 관절염이 있는 경우, 어떤 운동이 가장 효과적인가요?
# 관절염 통증 완화를 위해서는 어떤 약을 복용하는게 좋아? // gpt 
# 관절 부위가 아파 통증을 완화시키는 방법을 알려줘 // gpt 
# 무릎 관절에서 통증이 더 심해질 때는 언제야? 

########################
# 1. 임포트 
########################

# pip install requests beautifulsoup4

from urllib.parse import urlencode
import html
import re
import streamlit as st
import os
import io
import base64
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant as LangchainQdrant
from langchain.schema import Document
from openai import OpenAI as OpenAIClient
import logging
import time
import datetime
from streamlit_option_menu import option_menu
from pydub import AudioSegment
import requests
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import pandas as pd
from pydub.utils import which

########################
# 2. 설정 
########################
질병 = ['식중독', '관절염']  # 질병 목록 정의
의도 = ['원인', '증상', '예방', '정의', '치료', '진단']  # 의도 목록 정의
load_dotenv()

# st.set_page_config(layout="wide")
AudioSegment.converter = which("ffmpeg")
hospital_df = pd.read_csv('csv/병원.csv')
pharmacy_df = pd.read_csv('csv/약국.csv')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

# 로깅설정 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 전역 변수 및 클라이언트 초기화
COLLECTION_NAME = "son_son"
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vector_store = LangchainQdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
openai_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))

# 모델 초기화
@st.cache_resource
def load_model():
    model_name = "centwon/ko-gpt-trinity-1.2B-v0.5_v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model

# 모델 로드
tokenizer, model = load_model()

def TTS(api_key, api_url, text, voice, max_length=300):  # max_length를 API 제한에 맞게 설정
    headers = {
        "appKey": api_key,
        "Content-Type": "application/json"
    }

    # 텍스트를 최대 길이로 분할 (300자 기준)
    text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    audio_files = []

    for chunk in text_chunks:
        data = {
            "text": chunk,
            "lang": "ko-KR",
            "voice": voice,
            "speed": "0.8",
            "sr": "8000",
            "sformat": "wav"
        }
        response = requests.post(api_url, json=data, headers=headers)

        if response.status_code == 200:
            audio_data = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            audio_file = io.BytesIO()
            audio_data.export(audio_file, format="wav")
            audio_file.seek(0)
            audio_files.append(audio_file)
        else:
            st.error(f"API 요청 실패: {response.status_code}, {response.text}")
            return None

    # 여러 오디오 파일을 하나로 병합
    combined_audio = AudioSegment.empty()
    for audio in audio_files:
        audio.seek(0)
        combined_audio += AudioSegment.from_file(audio, format="wav")
    
    final_audio_file = io.BytesIO()
    combined_audio.export(final_audio_file, format="wav")
    final_audio_file.seek(0)
    return final_audio_file

################################################
# 3. 큐드란트 클래스
################################################


def google_search(query: str) -> str:
    """Google Custom Search API를 사용하여 첫 번째 검색 결과의 요약값만 반환합니다."""
    query_params = {
        'q': query,
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_SEARCH_ENGINE_ID
    }
    url = f"https://www.googleapis.com/customsearch/v1?{urlencode(query_params)}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"[google_search] API 요청 실패, 상태 코드: {response.status_code}")
            return "API 요청에 실패했습니다."

        results = response.json()
        logging.info(f"[google_search] API 응답: {results}")

        if 'items' not in results:
            return "검색 결과가 없습니다."

        # 첫 번째 검색 결과에서 필요한 정보 추출
        first_result = results['items'][0]
        title = first_result.get('title', '제목 없음')
        link = first_result.get('link', '링크 없음')
        snippet = first_result.get('snippet', '요약 없음')

        # 결과를 형식화된 문자열로 반환
        formatted_result = f"**{title}**\n{snippet}\n[자세히 보기]({link})"
        return formatted_result

    except Exception as e:
        logging.error(f"[google_search] 구글 검색 중 오류 발생: {str(e)}", exc_info=True)
        return "검색 중 오류가 발생했습니다. 다시 시도해 주세요."

def add_google_search_results(documents: List[Document], query: str) -> List[Document]:
    """Google 검색 결과를 Document 형태로 추가하는 함수"""
    snippet = google_search(query)
    if snippet:
        documents.append(Document(page_content=snippet, metadata={'source': 'Google Search'}))
    return documents

# class QdrantRetriever:
#     def __init__(self, client: QdrantClient, collection_name: str):
#         self.client = client
#         self.collection_name = collection_name

#     def retrieve(self, query_vector: List[float], top_k: int = 10) -> List[Document]:
#         try:
#             results = self.client.search(
#                 collection_name=self.collection_name,
#                 query_vector=query_vector,
#                 limit=top_k
#             )
#         except Exception as e:
#             logging.error(f"Qdrant 검색 중 오류 발생: {str(e)}", exc_info=True)
#             return []

#         if not results:
#             logging.info("Qdrant에서 유사한 쿼리가 없습니다.")
#             return []

#         documents = []
#         for result in results:
#             text = result.payload.get('답변', 'N/A')
#             metadata = {
#                 'id': result.id,
#                 '의도': result.payload.get('의도', 'N/A'),
#                 '질문': result.payload.get('질문', 'N/A'),
#                 '답변': result.payload.get('답변', 'N/A'),
#                 '질병': result.payload.get('질병', 'N/A'),
#                 'score': result.score  # 실제 Qdrant에서 가져온 유사도 값을 사용
#             }
#             documents.append(Document(page_content=text, metadata=metadata))

#         return documents



from typing import List
import logging
from qdrant_client import QdrantClient
# from some_library import Document  # Document 클래스가 있는 라이브러리를 임포트하세요.

class QdrantRetriever:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def retrieve(self, query: str, query_vector: List[float], 질병: Optional[str] = None, 의도: Optional[str] = None, top_k: int = 10) -> List[Document]:
        # 메타데이터 필터링 조건 설정
        filters = {"must": []}

        # 질병과 의도가 존재하는 경우 필터 조건에 추가
        if 질병:
            filters["must"].append({"key": "질병", "match": {"value": 질병}})
        if 의도:
            filters["must"].append({"key": "의도", "match": {"value": 의도}})

        try:
            if filters["must"]:  # 필터가 있을 경우에만 필터를 적용
                logging.info("메타데이터로 필터링에 성공했습니다.")  # 필터링 성공 로그 출력
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=filters  # 필터 조건 추가
                )
            else:  # 필터가 없을 경우 필터 없이 검색
                logging.info("질문자 쿼리에 메타데이터 정보가 없습니다.")  # 메타데이터 정보 없음 로그 출력
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )
        except Exception as e:
            logging.error(f"Qdrant 검색 중 오류 발생: {str(e)}", exc_info=True)
            return []

        # 검색 결과가 없을 경우 로그 출력
        if not results:
            logging.info("Qdrant에서 유사한 쿼리가 없습니다.")
            return []

        documents = []
        for result in results:
            text = result.payload.get('답변', 'N/A')
            metadata = {
                'id': result.id,
                '의도': result.payload.get('의도', 'N/A'),
                '질문': result.payload.get('질문', 'N/A'),
                '답변': result.payload.get('답변', 'N/A'),
                '질병': result.payload.get('질병', 'N/A'),
                'score': result.score
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents


def extract_metadata_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    사용자의 질문에서 질병과 의도를 추출합니다.
    질병과 의도를 반환합니다. 
    """
    질병_목록 = ['식중독', '관절염']
    의도_목록 = ['정의', '원인', '치료', '증상', '예방', '진단']

    # 질문에서 질병과 의도를 추출
    질병 = next((질 for 질 in 질병_목록 if 질 in query), None)
    의도 = next((의 for 의 in 의도_목록 if 의 in query), None)

    return 질병, 의도




def self_rag_search(query: str, threshold: float = 0.3) -> List[Document]:
    """Self RAG 적용: 모델이 검색할 지 여부를 결정"""

    # 사용자 질문에서 메타데이터 추출
    질병, 의도 = extract_metadata_from_query(query)
    
    # 쿼리 임베딩 생성
    query_vector = embeddings.embed_query(query)
    retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
    
    # retrieve 메서드 호출 시 질병과 의도를 필터링 조건으로 추가
    documents = retriever.retrieve(query, query_vector, 질병=질병, 의도=의도, top_k=5)

    # Qdrant와의 유사도 기준 적용
    filtered_documents = [doc for doc in documents if doc.metadata['score'] >= threshold]

    # 유사도가 낮을 경우 Google 검색을 수행하도록 설정
    if not filtered_documents:
        logging.info("[self_rag_search] 유사도 기준 미달, Google 검색 수행.")
        documents = add_google_search_results(documents, query)

    return filtered_documents or documents




def corrective_rag_search(query: str, threshold: float = 0.3) -> List[Document]:
    """Corrective RAG 적용: 검색 결과가 없거나 불만족스러울 경우 보완 검색"""
    query_vector = embeddings.embed_query(query)
    retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
    # 수정: retrieve 메서드 호출 시 query와 query_vector를 함께 전달합니다.
    documents = retriever.retrieve(query, query_vector, top_k=5)

    # Qdrant와의 유사도 기준 적용
    filtered_documents = [doc for doc in documents if doc.metadata['score'] >= threshold]

    # 검색 결과가 불만족스러울 경우 Google 검색 수행
    if not filtered_documents:
        logging.info("[corrective_rag_search] 기존 검색 결과 불만족, Google 검색 수행.")
        documents = add_google_search_results(filtered_documents, query)

    return filtered_documents or documents  # 빈 리스트가 아닌 항상 결과 반환



def adaptive_rag_search(query: str, feedback: Optional[Dict[str, float]] = None) -> List[Document]:
    """Adaptive RAG 적용: 피드백을 기반으로 검색 가중치 조정"""
    query_vector = embeddings.embed_query(query)
    retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
    # 수정: retrieve 메서드 호출 시 query와 query_vector를 함께 전달합니다.
    documents = retriever.retrieve(query, query_vector, top_k=5)

    if feedback:
        documents = adjust_weights_based_on_feedback(documents, feedback)

    return documents


# def ensemble_search(query: str, top_k: int = 5, bm25_weight: float = 0.5, vector_weight: float = 0.5) -> List[Document]:
#     """
#     BM25와 벡터 검색을 결합하여 앙상블 검색 수행
#     """
#     # BM25 검색 수행
#     bm25_scores = bm25.get_scores(query.split())
#     bm25_results = [Document(page_content=doc, metadata={'score': score, 'source': 'BM25'})
#                     for score, doc in sorted(zip(bm25_scores, documents), reverse=True) if score > 0]

#     # 벡터 검색 수행
#     query_vector = embeddings.embed_query(query)
#     retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
#     vector_results = retriever.retrieve(query_vector, top_k=top_k)

#     # BM25와 벡터 검색 결과 결합
#     combined_results = bm25_results + [doc for doc in vector_results if doc.page_content not in [bm25_doc.page_content for bm25_doc in bm25_results]]

#     # 가중치 점수를 기반으로 결과 정렬
#     weighted_results = sorted(combined_results, key=lambda x: calculate_weighted_score(x, bm25_results, vector_results, bm25_weight, vector_weight))

#     return weighted_results

def search_documents(query: str) -> List[Document]:
    """
    벡터 검색을 통해 문서를 검색합니다. 결과가 없을 경우 Google 검색을 수행합니다.
    """
    try:
        # 벡터 검색 수행
        query_vector = embeddings.embed_query(query)
        retriever = QdrantRetriever(client=client, collection_name=COLLECTION_NAME)
        # 수정: retrieve 메서드 호출 시 query와 query_vector를 함께 전달합니다.
        documents = retriever.retrieve(query, query_vector, top_k=5)

        # 검색 결과가 없을 경우 Google 검색 수행
        if not documents:
            logging.warning("[search_documents] 검색 결과가 없습니다. Google 검색을 수행합니다.")
            documents = add_google_search_results([], query)
            st.write(documents[0].page_content)  # Google 검색 결과 출력
        else:
            logging.info("[search_documents] 검색 결과를 성공적으로 가져왔습니다.")

        return documents

    except Exception as e:
        logging.error(f"[search_documents] 검색 중 오류 발생: {str(e)}", exc_info=True)
        return []

########################
# 3. LLM
########################

def generate_gpt4_response(query: str, detailed: bool = False) -> Tuple[str, str, float]:
    """
    GPT-4 응답 생성 함수
    기본 응답 또는 자세한 응답을 생성
    """
    try:
        # Self RAG 검색 수행
        documents = self_rag_search(query, threshold=0.3)
        if not documents:
            return "유사한 답변이 없습니다. 다시 한번 질문해주겠습니까?", "GPT-4", 0.0

        # 검색된 문서에서 메타데이터 추출
        top_document = documents[0]
        metadata = {
            '질병': top_document.metadata.get('질병', 'N/A'),
            '의도': top_document.metadata.get('의도', 'N/A'),
            '질문': top_document.metadata.get('질문', 'N/A'),
            '답변': top_document.metadata.get('답변', 'N/A'),
            '유사도': top_document.metadata.get('score', 'N/A')
        }

        # 프롬프트 설정
        if detailed:
            prompt_length = (
                f"500글자 안의 문장으로 상세하게 설명하세요.\n"
                f"필요한 경우 추가 검사나 후속 조치에 대해 언급하세요.\n"
                f"예방 및 치료방법에 대한 조언을 포함하세요\n"
            )
        else:
            prompt_length = "300글자 안의 문장으로 간단하게 설명하세요"

        system_message = (
            f"{metadata.get('질병', 'N/A')}와 관련된 주제에 집중해야 합니다.\n"
            f"{metadata.get('답변', 'N/A')}을 바탕으로 적절한 의학적 답변을 제공해야 합니다.\n"
            f"증상에 대한 설명, 원인, 응급처치법을 요약해서 주세요.\n"
            f"{prompt_length}"
        )

        user_message = (
            f"사용자 질문: {query}\n\n"
            f"참고 답변: {metadata.get('답변', 'N/A')}\n"
            f"{metadata.get('질병', 'N/A')}에 대해 다음 사항을 포함하여 상세하고 체계적인 응답을 제공해 주세요."
        )

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            n=1,
        )

        generated_response = response.choices[0].message.content.strip()
        processing_time = time.time() - time.time()
        logging.info(f"GPT-4 응답 생성 완료: 길이={len(generated_response)}")
        
        return generated_response, "GPT-4", processing_time

    except Exception as e:
        logging.error(f"GPT-4 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. GPT-4 응답을 생성하는 중에 오류가 발생했습니다.", "GPT-4", 0.0

################################################
# 3-2. 파인튜닝 모델 
################################################
def generate_custom_model_response(query: str, max_tokens: int = 75) -> Tuple[str, str, float]:
    try:
        # 문서 검색
        documents = self_rag_search(query)
        if not documents:
            return "유사한 답변이 없습니다. 다시 한번 질문해주겠습니까?", "Custom Model", 0.0

        top_document = documents[0]
        metadata = {
            '질문': top_document.metadata.get('질문', 'N/A'),
            '의도': top_document.metadata.get('의도', 'N/A'),
            '답변': top_document.metadata.get('답변', 'N/A'),
            '질병': top_document.metadata.get('질병', 'N/A'),
            '유사도': top_document.metadata.get('score', 'N/A')
        }

        if max_tokens > 120:
            prompt = (
                f"You are a language model that aims to provide a detailed and comprehensive explanation while retaining the original content.\n"
                f"Take the following sentence and explain it in detail, providing additional context or clarification as needed while keeping the original meaning intact:\n"
                f"Original sentence: <{metadata.get('답변', 'N/A')}>\n"
                "Ensure that the explanation is thorough and includes relevant information to help the reader fully understand the topic.\n"
                f"Your response should only focus on the topic related to [food poisoning].\n"
                "Generate a detailed response in Korean:\n"
                "---\n"
                "답변임!!:"
            )
        else:
            prompt = (
                f"**답변**: [{metadata.get('답변', 'N/A')}]\n"
                f"위 답변을 바탕으로, {metadata.get('질병', 'N/A')}에 대한 요약된 내용을 작성하세요. "
                "답변은 한국어로 작성되어야 하세요\n"
                "답변임!!:"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start_time = time.time()

        bad_words_ids_ko = tokenizer(["식도염"], add_special_tokens=False).input_ids
        bad_words_ids_en = tokenizer(["esophagitis"], add_special_tokens=False).input_ids

        bad_words_ids = bad_words_ids_ko + bad_words_ids_en

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            temperature=0.4,
            do_sample=False,
            early_stopping=True,
            bad_words_ids=bad_words_ids,
            repetition_penalty = 1.1   
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
     
        if not response.endswith('.'):
            last_period_index = response.rfind('.')
            if last_period_index != -1:
                response = response[:last_period_index + 1]  # 마지막 마침표까지의 부분만 남김

        logging.info("파인튜닝된 모델을 사용 중입니다.")
        logging.info('-'*30)
        logging.info('참고한 메타데이터 정보')
        logging.info('-'*30)
        logging.info(query)
        logging.info(f"질병 : {metadata.get('질병', 'N/A')}")
        logging.info(f"의도 : {metadata.get('의도', 'N/A')}")
        logging.info(f"유사한 질문 : {metadata.get('질문', 'N/A')}")
        logging.info(f"참고한 답변 : {metadata.get('답변', 'N/A')}")
        logging.info(f"유사도 (score): {metadata.get('유사도', 'N/A')}")

        if "답변임!!:" in response:
            response = response.split("답변임!!:")[-1].strip()    
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()

        return response, "Custom Model", processing_time

    except Exception as e:
        logging.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", "Custom Model", 0.0

########################
# 피드백 관련 함수들
########################

def collect_user_feedback(documents: List[Document]) -> Dict[str, float]:
    """사용자로부터 피드백을 수집합니다. 슬라이더를 움직인 경우에만 피드백을 받습니다."""
    if documents:
        # 검색된 문서 중 첫 번째 문서를 선택
        doc = documents[0]
        st.write("검색 결과에 대해 피드백을 주세요:")
        
        # 메타데이터에 '질문' 키가 있는지 확인하고 기본값 설정
        question_text = doc.metadata.get('질문', '해당 질문이 없습니다.')

        # 슬라이더 초기 값을 중립으로 설정
        initial_value = 1
        score = st.slider(f"{question_text}의 질문과 유사하다고 생각하시나요?", min_value=0, max_value=3, value=initial_value)

        # 슬라이더 값이 변경되었는지 확인
        if score != initial_value:  # 슬라이더가 움직였을 때만 피드백 생성
            feedback = {doc.metadata['id']: score}
            return feedback
        else:
            return {}  # 슬라이더가 움직이지 않으면 빈 딕셔너리 반환
    else:
        st.write("피드백을 제공할 문서가 없습니다.")
        return {}

def adjust_weights_based_on_feedback(documents: List[Document], feedback: Optional[Dict[str, float]] = None) -> List[Document]:
    """사용자 피드백을 기반으로 가중치를 조정합니다. 피드백이 없는 경우 조정하지 않습니다."""
    if feedback:
        for doc in documents:
            doc_id = doc.metadata['id']
            if doc_id in feedback:
                doc.metadata['score'] += feedback[doc_id]
                logging.info(f"문서 {doc_id}의 가중치를 {feedback[doc_id]}만큼 조정.")
    return documents

########################
# 5. 스트림릿 페이지 
########################
# 기본 배경 이미지 설정 함수
def basic_background(png_file, opacity=1):
   bin_str = get_base64(png_file)
   page_bg_img = f"""
   <style>
   .stApp {{
       background-image: url("data:image/png;base64,{bin_str}");
       background-size: cover;  /* 이미지가 잘리지 않도록 설정 */
       background-position: center;
       background-repeat: no-repeat;
       width: 100%;
       height: 100vh;
       opacity: {opacity};
       z-index: 0;
       position: fixed;
   }}
   </style>
   """
   st.markdown(page_bg_img, unsafe_allow_html=True)
   

# 이미지를 base64 문자열로 변환하는 함수
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 글자 하나씩 출력하는 함수
def display_typing_effect(text):
    output = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        output.markdown(f"<p>{displayed_text}</p>", unsafe_allow_html=True)
        time.sleep(0.05)

def play_tts(api_key, tts_url, text, voice):
    """TTS 재생을 위한 함수"""
    if api_key and voice and text:  # 모든 매개변수가 유효한 경우에만 TTS 실행
        try:
            tts_audio = TTS(api_key, tts_url, text, voice)
            if tts_audio:
                audio_bytes = tts_audio.read()
                encoded_audio = base64.b64encode(audio_bytes).decode()
                audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"TTS 에러: {e}")



# 나머지 Streamlit 부분 유지
# Chatbot 인터페이스 및 UI 구성 부분

def sidebar_menu():
    if st.session_state.page != "main":
        with st.sidebar:
            choice = option_menu(
                "Menu", ["챗봇", "병원&약국", "응급상황대처법"],
                icons=['bi bi-robot', 'bi bi-capsule', ''],
                menu_icon="app-indicator", default_index=0,
                styles={
                    "container": {"padding": "4!important", "background-color": "#fafafa"},
                    "icon": {"color": "black", "font-size": "25px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#fafafa"},
                    "nav-link-selected": {"background-color": "#08c7b4"},
                }
            )
            if choice == "챗봇":
                st.session_state.page = "chat_interface"
            elif choice == "병원&약국":
                st.session_state.page = "hospital_pharmacy"
            elif choice == "응급상황대처법":
                st.session_state.page = "video"

def main():
    # 세션 상태 초기화
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.page = "main"
        st.session_state.model_option = "GPT-4"  # 기본값을 "GPT-4"로 설정
        st.session_state.gpt_usage_count = 3  # GPT 모델 초기 사용 가능 횟수 설정
        st.session_state.chat_history = []

    # 메인 페이지가 아닐 때만 사이드바 메뉴 호출
    if st.session_state.page != "main":
        sidebar_menu()

    # 페이지 상태에 따른 페이지 호출
    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "chat_interface":
        chat_interface()
    elif st.session_state.page == "hospital_pharmacy":
        hospital_pharmacy_page()
    elif st.session_state.page == "video":
        video()
    elif st.session_state.page == "ad_page":
        ad_page()

def main_page():
    background_image = '사진/효자손.png'
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: #1187cf;
            background-image: url("data:image/png;base64,{get_base64(background_image)}");
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            width: 100%;
            height: 100vh;
            position: fixed;
            z-index: 0;
        }}
        div.stButton > button {{
            position: fixed;
            bottom: 450px;
            left: 50%;
            transform: translateX(-50%);
            background-color:  #FFFFFF;
            color: black;
            padding: 15px 32px;
            font-size: 80px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            z-index: 10;
        }}
        div.stButton > button:hover {{
            background-color: #007BFF;
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

    if st.button("효자SON 이용하기 :point_right: "):
        st.session_state.page = "chat_interface"

def reduce_usage_count():
    if st.session_state.model_option == "GPT-4":
        if st.session_state.gpt_usage_count > 0:
            st.session_state.gpt_usage_count -= 1
        else:
            show_usage_alert()

def play_tts_warning(api_key, warning_text):
    tts_audio = TTS(api_key, os.getenv('TTS_URL'), warning_text, "juwon")
    if tts_audio:
        audio_bytes = tts_audio.read()
        encoded_audio = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

def request_and_handle_feedback():
    st.feedback(
        options="thumbs",
        key="feedback",
        on_change=lambda: st.success("피드백 감사합니다!")
    )

def show_usage_alert():
    api_key = os.getenv('TTS_API_KEY')
    if st.session_state.gpt_usage_count <= 0:
        warning_text = "약과가 다 떨어졌어요!\n약과를 충전하시겠나요?\n아니면 제 동생을 불러드릴까요?"
        with st.chat_message("assistant", avatar="사진/아바타2.png"):
            display_typing_effect(warning_text)
        if api_key:
            play_tts_warning(api_key, warning_text)
        display_typing_effect("GPT 모델 사용이 끝났습니다. 다음 대화를 평가해 주세요:")
        request_and_handle_feedback()

def display_remaining_count_after_response():
    """GPT-4 모델이 답변을 생성한 후 남은 약과 개수를 대화 형식으로 출력하는 함수"""
    if st.session_state.get('model_option') == "GPT-4":
        remaining_count = st.session_state.get('gpt_usage_count', 0)
        # 남은 약과 개수를 대화 형식으로 출력
        with st.chat_message("assistant", avatar="사진/약과.png"):
            st.write(f"약과가 {remaining_count}개 남았어요!")
###################################            
###################################
def chat_interface():
    background_image = '사진/002.png'
    basic_background(background_image)

    # 세션 상태 초기화
    if 'gpt_usage_count' not in st.session_state:
        st.session_state.gpt_usage_count = 3  # 사용 가능 횟수 초기화

    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = False

    if 'follow_up_question' not in st.session_state:
        st.session_state.follow_up_question = None

    if 'selected_disease' not in st.session_state:
        st.session_state.selected_disease = None

    st.markdown("""
        <style>
        .stTitle, .stButton button, .stRadio label, .stChatMessage p {
            font-size: 25px !important;
        }
        .stRadio label {
            font-size: 40px !important;
        }
        .custom-text {
            font-size: 30px !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    api_key = os.getenv('TTS_API_KEY')
    tts_url = os.getenv('TTS_URL')

    col1, col2 = st.columns(2)

    # 모델 선택 및 음성 안내 설정
    with col1:
        st.subheader("효자 선택하기")
        selected_model_nickname = st.radio(" ", ("귀여운 막내", "든든한 맏형(유료)"), index=0)
    with col2:
        st.subheader("음성안내 선택")
        tts_enabled = st.radio(" ", ("사용", "사용안함"), index=1)
        st.session_state.tts_enabled = (tts_enabled == "사용")

    # 모델에 따라 avatar 및 voice 설정
    if selected_model_nickname == "든든한 맏형(유료)":
        selected_model = "GPT-4"
        st.session_state.voice = 'juwon'
        st.session_state.avatar = "사진/아바타2.png"
    else:
        selected_model = "Custom Model"
        st.session_state.voice = 'doyun'
        st.session_state.avatar = "사진/아바타1.png"

    assistant_avatar = st.session_state.avatar
    assistant_voice = st.session_state.voice

    if 'model_option' not in st.session_state or st.session_state.model_option != selected_model:
        st.session_state.model_option = selected_model
        if selected_model == "Custom Model":
            avatar_image = "사진/아바타1.png"
            initial_message = "귀염둥이 막내! 간단한 답변은 제가 설명드릴게요!"
        else:
            avatar_image = "사진/아바타2.png"
            initial_message = "든든한 맏형! 자세한 내용이 필요하시다면 제가 설명드릴게요!"

        with st.chat_message("assistant", avatar=avatar_image):
            display_typing_effect(initial_message)

        if st.session_state.tts_enabled:
            play_tts(api_key, tts_url, initial_message, st.session_state.voice)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "무엇을 도와드릴까요?", "avatar": "사진/아바타2.png"}]
        with st.chat_message("assistant", avatar="사진/아바타2.png"):
            st.write("무엇을 도와드릴까요?")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    user_input = st.chat_input("예시 문장 : 복통 설사 구토등의 증상이 있어 무슨병일까?")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input, "avatar": "사진/user.png"})
        with st.chat_message("user", avatar="사진/user.png"):
            display_typing_effect(user_input)

        with st.spinner("답변을 생성 중입니다. 잠시만 기다려 주세요..."):
            if st.session_state.model_option == "GPT-4":  
                response, model, processing_time = generate_gpt4_response(user_input, detailed=False)
                assistant_avatar = "사진/아바타2.png"
                assistant_voice = "juwon"
            else:
                response, model, processing_time = generate_custom_model_response(user_input)
                assistant_avatar = "사진/아바타1.png"
                assistant_voice = "doyun"

            formatted_response = response
            st.session_state.chat_history.append({"role": "assistant", "content": formatted_response, "avatar": assistant_avatar})
            with st.chat_message("assistant", avatar=assistant_avatar):
                display_typing_effect(formatted_response)
                if st.session_state.tts_enabled:
                    play_tts(api_key, tts_url, formatted_response, assistant_voice)
            st.session_state.selected_intent = None  
        reduce_usage_count()
        reduce_usage_count()

    # 질병이 선택된 경우 추가 질문 선택
    if st.session_state.selected_disease:
        st.markdown(f"<p class='custom-text'>{st.session_state.selected_disease}에 대한 다른 게 궁금하신가요?</p>", unsafe_allow_html=True)

        # 사용자 선택 상태를 세션에서 관리
        if 'selected_intent' not in st.session_state:
            st.session_state.selected_intent = None  # 처음 실행 시 초기화
        
        # 동적 key를 사용하여 항상 새로운 selectbox로 렌더링
        dynamic_key = f"intent_selectbox_{st.session_state.selected_disease}_{len(st.session_state.chat_history)}"  # 수정됨!!

        # Selectbox로 사용자에게 질문 의도를 선택하게 함, 기본값으로 None 추가
        selected_intent = st.selectbox(
            " ", 
            [None] + 의도,  # 기본값으로 None 추가
            index=0,  # 항상 None을 기본값으로 설정
            format_func=lambda x: '질문을 선택하세요' if x is None else x,
            key=dynamic_key  # 동적 키 사용 # 수정됨!!
        )

        # 실제 사용자가 선택한 경우에만 처리
        if selected_intent is not None:
            # 새로운 쿼리 생성
            new_query = f"{st.session_state.selected_disease}의 {selected_intent}이 궁금해"

        # 로딩 메시지 추가

            with st.spinner("답변을 생성 중입니다. 잠시만 기다려 주세요..."):  # 수정됨!!
                # 커스텀 모델로 응답 생성
                response, model, processing_time = generate_custom_model_response(new_query)
                
                # 응답을 대화 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": assistant_avatar})
                
                # 응답을 대화 창처럼 출력
                with st.chat_message("assistant", avatar=assistant_avatar):
                    display_typing_effect(response)
                    if st.session_state.tts_enabled:
                        play_tts(api_key, tts_url, response, assistant_voice)

            # 답변 출력 후 선택 상태를 기본값(None)으로 재설정
            st.session_state.selected_intent = None  # 선택 초기화 # 수정됨!!

    # 남은 약과 개수 출력
    display_remaining_count_after_response()





def hospital_pharmacy_page():
    # 세션 상태에 병원과 약국 데이터를 한 번만 로드합니다.
    if 'hospital_data' not in st.session_state:
        st.session_state['hospital_data'] = pd.read_csv('csv/병원.csv')
    if 'pharmacy_data' not in st.session_state:
        st.session_state['pharmacy_data'] = pd.read_csv('csv/약국.csv')

    # 세션 상태에서 데이터 가져오기
    hospital = st.session_state['hospital_data']
    pharmacy = st.session_state['pharmacy_data']

    # 좌표 데이터를 숫자(float) 형식으로 변환
    hospital['좌표(Y)'] = pd.to_numeric(hospital['좌표(Y)'], errors='coerce')
    hospital['좌표(X)'] = pd.to_numeric(hospital['좌표(X)'], errors='coerce')
    pharmacy['좌표(Y)'] = pd.to_numeric(pharmacy['좌표(Y)'], errors='coerce')
    pharmacy['좌표(X)'] = pd.to_numeric(pharmacy['좌표(X)'], errors='coerce')

    # 지도 객체를 생성합니다.
    my_map = folium.Map(
        location=[35.1614592, 129.1625655], 
        zoom_start=16
    )

    # 병원 마커를 지도에 추가하는 함수
    def add_hospital_markers(map_obj):
        for idx, row in hospital.iterrows():
            folium.Marker(
                location=[row['좌표(Y)'], row['좌표(X)']],
                popup=folium.Popup(
                    f"{row['요양기관명']}<br>",
                    max_width=450
                ),
                icon=folium.Icon(color='pink', icon='hospital', prefix='fa'),
                tooltip=row['요양기관명']
            ).add_to(map_obj)

    # 약국 마커를 지도에 추가하는 함수
    def add_pharmacy_markers(map_obj):
        for idx, row in pharmacy.iterrows():
            folium.Marker(
                location=[row['좌표(Y)'], row['좌표(X)']],
                popup=folium.Popup(
                    f"{row['요양기관명']}<br>",
                    max_width=450
                ),
                icon=folium.Icon(color='blue', icon='info-sign'),
                tooltip=row['요양기관명']
            ).add_to(map_obj)

    # Streamlit 페이지 설정
    st.title('주변 병원 & 약국 정보를 알려드릴께요! ')

    # 사용자 선택 상자 추가
    option = st.selectbox(
        '어떤 곳을 찾고 싶으신가요?',
        ('전체', '병원', '약국')
    )

    # 사용자의 선택에 따라 마커를 지도에 추가
    if option == '병원':
        add_hospital_markers(my_map)
    elif option == '약국':
        add_pharmacy_markers(my_map)
    else:
        add_hospital_markers(my_map)
        add_pharmacy_markers(my_map)

    # Folium 맵을 Streamlit에 표시
    output = st_folium(my_map, width=800, height=600)

    # 클릭된 위치 확인
    if output and 'last_object_clicked' in output:  # 클릭 이벤트가 있는지 확인
        clicked_location = output['last_object_clicked']
        if clicked_location:
            clicked_lat = clicked_location['lat']
            clicked_lng = clicked_location['lng']

            # 병원 데이터에서 정확한 좌표로 검색
            hospital_selected = hospital[
                (hospital['좌표(Y)'] == clicked_lat) & 
                (hospital['좌표(X)'] == clicked_lng)
            ]

            # 약국 데이터에서 정확한 좌표로 검색
            pharmacy_selected = pharmacy[
                (pharmacy['좌표(Y)'] == clicked_lat) & 
                (pharmacy['좌표(X)'] == clicked_lng)
            ]

            # 병원 정보 출력
            if not hospital_selected.empty:
                # 병원 이름이 "약국"으로 끝나는지 확인하고 출력하지 않음
                if not hospital_selected.iloc[0]['요양기관명'].endswith("약국"):
                    st.header(f"**병원:** {hospital_selected.iloc[0]['요양기관명']}")
                    st.subheader(f"**병원 종류:** {hospital_selected.iloc[0]['종별코드명']}")
                    st.subheader(f"**주소:** {hospital_selected.iloc[0]['주소']}")
                    st.subheader(f"**전화번호:** {hospital_selected.iloc[0]['전화번호']}")

            # 약국 정보 출력
            if not pharmacy_selected.empty:
                st.header(f"**약국:** {pharmacy_selected.iloc[0]['요양기관명']}")
                st.subheader(f"**주소:** {pharmacy_selected.iloc[0]['주소']}")
                st.subheader(f"**전화번호:** {pharmacy_selected.iloc[0]['전화번호']}")

    # 마지막 페이지 상태 저장
    st.session_state['last_page'] = 'hospital_pharmacy_page'





def video():
    st.header("응급상황 대처 방법")
    st.video('https://www.youtube.com/watch?v=HktGyea8zcw')

    st.header("식중독 대처방법")
    st.video('https://www.youtube.com/watch?v=nyC11uFLD28')

    st.header("관절염 환자의 팁")
    st.video('https://www.youtube.com/watch?v=75rWlyi9lXU&t=1s')

    st.header("응급상황 대처방법")
    st.video('https://www.youtube.com/watch?v=k3DVIXXmmA0')

def ad_page():
    background_image = '사진/광고페이지.png'
    basic_background(background_image)
    st.session_state.usage_count = 1
    st.session_state.page = "chat_interface"
    st.success('광고 시청으로 횟수가 1회 충전되었습니다.')
    st.experimental_rerun()

if __name__ == "__main__":
    main()
