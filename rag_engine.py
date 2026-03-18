import os
import re
from dotenv import load_dotenv

load_dotenv()
from typing import List, Dict, Optional, AsyncGenerator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
import asyncio
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        # Fallback: use simple ensemble
        class EnsembleRetriever:
            def __init__(self, retrievers, weights):
                self.retrievers = retrievers
                self.weights = weights
            
            def invoke(self, query):
                all_docs = []
                for retriever in self.retrievers:
                    all_docs.extend(retriever.invoke(query))
                return all_docs[:10]  # Return top 10

class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder model."""
        print(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("✓ Model loaded successfully!")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User's search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Top_k reranked documents with scores
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by scores (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top_k documents
        reranked_docs = [doc for doc, score in scored_docs[:top_k]]
        
        # Add scores to metadata
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            doc.metadata['rerank_score'] = float(score)
            doc.metadata['rerank_position'] = i + 1
        
        return reranked_docs
class RAGEngine:
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        llm_model: str = "Qwen/Qwen2.5-72B-Instruct"
    ):
        self.pdf_path = pdf_path or os.environ.get("PDF_PATH", "faqdata.pdf")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.embeddings = None
        self.vector_store = None
        self.reranker = None
        self.chain = None
        self.documents = None
        self.reranking_retriever = None
        self.hf_client = None

    async def initialize(self):
        """Initialize all RAG components"""
        print("  🔧 Initializing RAG Engine components...")
        self.chain = None  # Ensure state is reset for initialization attempt
        
        # 1. Load documents
        if not os.path.exists(self.pdf_path):
            print(f"  ❌ ERROR: PDF not found at {self.pdf_path}")
            raise FileNotFoundError(f"PDF knowledge base not found at {self.pdf_path}. Please set PDF_PATH or ensure faqdata.pdf exists.")
            
        print(f"  📄 Loading documents from: {self.pdf_path}")
        await self._load_documents()
        
        # 2. Initialize embeddings
        print(f"  🎯 Loading embeddings: {self.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 3. Create vector store
        print("  💾 Creating vector store...")
        await self._create_vector_store()
        
        # 4. Initialize reranker
        print("  🔀 Initializing reranker...")
        self.reranker = CrossEncoderReranker()
        
        # 5. Initialize LLM
        print(f"  🤖 Initializing LLM: {self.llm_model}")
        if not self.hf_token:
            print("  ❌ ERROR: HF_TOKEN is missing from environment")
            # Clear partially set instance if applicable
            self.hf_client = None
            raise ValueError("HF_TOKEN not found. Set it in environment secrets (HF Spaces) or .env file.")
        
        self.hf_client = InferenceClient(
            model=self.llm_model,
            token=self.hf_token
        )
        
        # 6. Build RAG chain
        print("  ⛓️ Building RAG chain...")
        await self._build_chain()
        
        print("  ✅ RAG Engine initialized successfully!")

    async def _load_documents(self):
        """Load and split PDF documents"""
        # Check if PDF exists
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = text_splitter.split_documents(docs)
        print(f"    ✓ Loaded {len(docs)} pages, split into {len(self.documents)} chunks")
    
    async def _create_vector_store(self):
        if self.documents:
            self.vector_store = await asyncio.to_thread(
                Chroma.from_documents,
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory="chroma_db",
                collection_name="faq_docs"
            )
        else:
            self.vector_store = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embeddings,
                collection_name="faq_docs"
            )

    
    async def _build_chain(self):
        """Build the RAG chain with hybrid search + reranking"""
        
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={'k': 10}
        )

        # 👉 BM25 only if documents exist
        if self.documents:
            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = 10

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
        else:
            hybrid_retriever = vector_retriever

        
        # Reranking retriever
        def retrieve_and_rerank(query: str) -> List[Document]:
            docs = hybrid_retriever.invoke(query)
            return self.reranker.rerank(query, docs, top_k=3)
        
        reranking_retriever = RunnableLambda(retrieve_and_rerank)
        
        # Store retriever for later access
        self.reranking_retriever = reranking_retriever
        
        # Format documents
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)
        prompt = PromptTemplate(
                template="""Using ONLY the policy context below, draft a professional email reply.

        --- POLICY CONTEXT ---
        {context}
        --- END CONTEXT ---

        Customer Name: {customer_name}
        Email Subject: {email_subject}
        Customer Question: {question}

        Write the reply now.""",
            input_variables=['context', 'question', 'customer_name', 'email_subject']
        )

        SYSTEM_PROMPT = """You are an automated email support assistant for an e-commerce company. Your role is to respond to customer emails using ONLY the policy information provided in the context.

        IDENTITY & ROLE:
        - You handle inquiries about returns, refunds, shipping, orders, account management, and policies.
        - Always sign off as "Customer Support Team."

        RESPONSE FORMAT:
        - Always reply in proper email format:
        - Subject line prefixed with "Re:"
        - Greeting using the customer's name (if provided; otherwise use "Hi there")
        - Concise body answering the question directly
        - Sign off as "Customer Support Team"
        - Use bullet points or numbered steps ONLY when listing multiple items or steps.
        - LENGTH: Keep responses short. 3–6 sentences for simple questions. No filler phrases like "Great question!" or "I'd be happy to help."

        CORE RULES:
        1. ONLY use information from the provided context. Do NOT fabricate, guess, or assume any policy, timeline, or fee.
        2. Quote specific numbers and timeframes accurately (e.g., "30 days", "$7.99").
        3. ESCALATION: If the question is NOT covered in the context, or requires account investigation, billing disputes, or technical troubleshooting — do NOT say "I don't know." Instead write: "For this, please reach out to our support team directly" and include whatever contact details are available in the context.
        4. TONE: Warm and professional. Acknowledge the concern in one sentence, then answer directly.
        5. For multi-step processes, use numbered steps only.
        6. If an order number is mentioned, acknowledge it and direct them to their account dashboard for live status.
        7. For multiple questions, use a numbered list with one direct answer per question."""
        # LLM wrapper as a RunnableLambda
        def call_llm(prompt_value) -> str:
            """Call HuggingFace InferenceClient for chat completion."""
            prompt_text = prompt_value.to_string() if hasattr(prompt_value, 'to_string') else str(prompt_value)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ]
            
            try:
                response = self.hf_client.chat_completion(messages, max_tokens=1500, temperature=0.1)
                
                # Extract content from response
                try:
                    content = response.choices[0].message["content"]
                except (KeyError, TypeError):
                    content = response.choices[0].message.content
                
                return content
            except Exception as e:
                print(f"Error calling LLM: {str(e)}")
                return f"I'm sorry, I'm having trouble generating a response right now. Please try again later or contact support."



        self.chain = (
            RunnableParallel({
                'context': RunnableLambda(lambda x: x['query']) | reranking_retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            | RunnableLambda(lambda x: {
                'context': x['context'],
                'question': x['question']['query'],
                'customer_name': x['question'].get('customer_name', 'Valued Customer'),
                'email_subject': x['question'].get('email_subject', 'Your Inquiry')
            })
            | prompt
            | RunnableLambda(call_llm)
        )

    async def get_response(
        self,
        query: str,
        customer_name: str = "Valued Customer",
        email_subject: str = "Your Inquiry",
        history: Optional[List] = None,
        category: Optional[str] = None
    ) -> Dict:
        """
        Get response from RAG system
        
        Args:
            query: User's question
            customer_name: Name of the customer for email greeting
            email_subject: Subject line of the customer's email
            history: Chat history (not used currently)
            category: Optional category filter
            
        Returns:
            Dict with answer, docs, and metadata
        """
        if not self.chain:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")
        
        # Build input dict with metadata for the chain
        chain_input = {
            'query': query,
            'customer_name': customer_name,
            'email_subject': email_subject
        }
        
        # Get response
        answer = await asyncio.to_thread(self.chain.invoke, chain_input)
        
        # Get retrieved documents for transparency
        docs = await asyncio.to_thread(self.reranking_retriever.invoke, query)
        
        # Calculate confidence (based on rerank scores)
        confidence = None
        if docs and 'rerank_score' in docs[0].metadata:
            confidence = float(docs[0].metadata['rerank_score'])
        
        # Format docs for response
        doc_list = [
            {
                "content": doc.page_content[:200] + "...",
                "score": doc.metadata.get('rerank_score', 0.0),
                "source": doc.metadata.get('source', 'unknown')
            }
            for doc in docs
        ]
        
        return {
            "answer": answer,
            "docs": doc_list,
            "confidence": confidence,
            "category": category
        }
    async def get_streaming_response(
        self,
        query: str,
        history: Optional[List] = None,
        category: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Get streaming response (token by token)
        
        Args:
            query: User's question
            history: Chat history
            category: Optional category filter
            
        Yields:
            Response chunks
        """
        # For now, simulate streaming by chunking the response
        # In production, use LangChain's streaming capabilities
        result = await self.get_response(query, category=category)
        answer = result["answer"]
        
        # Split into words and yield
        words = answer.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05)  # Simulate typing delay
    
    def load_vector_store(self):
        """Load existing Chroma DB"""
        self.vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings,
            collection_name="faq_docs"
        )
        print("✓ Chroma vector store loaded")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def test_rag_engine():
    """Test function for RAG engine"""
    print("🧪 Testing RAG Engine...")
    
    # Initialize
    engine = RAGEngine()
    await engine.initialize()
    
    # Test queries
    test_queries = [
        {"query": "What is the refund processing time?", "name": "John", "subject": "Refund Timeline"},
        {"query": "What is the return policy during holiday season?", "name": "Sarah", "subject": "Holiday Returns"},
        {"query": "How do I return a damaged item?", "name": "Mike", "subject": "Damaged Product Return"}
    ]
    
    for test in test_queries:
        print(f"\n❓ Query: {test['query']}")
        result = await engine.get_response(
            query=test['query'],
            customer_name=test['name'],
            email_subject=test['subject']
        )
        print(f"💬 Answer:\n{result['answer']}")
        print(f"📊 Confidence: {result['confidence']}")
        print("-" * 80)

if __name__ == "__main__":
    # Test the engine
    asyncio.run(test_rag_engine())

