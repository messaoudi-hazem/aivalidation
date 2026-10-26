import sys
sys.dont_write_bytecode = True

from typing import List
import re
from pydantic import BaseModel, Field

from langchain_core.tools import tool


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
  """
  List of IDs of the applicants to retrieve resumes for
  """
  id_list: List[str] = Field(..., description="List of IDs of the applicants to retrieve resumes for")

class JobDescription(BaseModel):
  """
  Descriptions of a job to retrieve similar resumes for
  """
  job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for") 



class RAGRetriever():
  def __init__(self, vectorstore_db, df):
    self.vectorstore = vectorstore_db
    self.df = df

  def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
    fused_scores = {}
    for doc_list in document_rank_list:
      for rank, (doc, _) in enumerate(doc_list.items()):
        if doc not in fused_scores:
          fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

  def __retrieve_docs_id__(self, question: str, k=50):
    docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
    docs_score = {str(doc.metadata["ID"]): score for doc, score in docs_score}
    return docs_score

  def retrieve_id_and_rerank(self, subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
      document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
    reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
    return reranked_documents

  def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
    id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
    retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
    retrieved_documents = [id_resume_dict[id] for id in retrieved_ids]
    for i in range(len(retrieved_documents)):
      retrieved_documents[i] = "Applicant ID " + retrieved_ids[i] + "\n" + retrieved_documents[i]
    return retrieved_documents 
   


class SelfQueryRetriever(RAGRetriever):
  def __init__(self, vectorstore_db, df):
    super().__init__(vectorstore_db, df)

    self.meta_data = {
      "rag_mode": "",
      "query_type": "no_retrieve",
      "extracted_input": "",
      "subquestion_list": [],
      "retrieved_docs_with_scores": []
    }

  def retrieve_docs(self, question: str, llm, rag_mode: str):
    @tool(args_schema=ApplicantID)
    def retrieve_applicant_id(id_list: list):
      """Retrieve resumes for applicants in the id_list"""
      retrieved_resumes = []

      for id in id_list:
        try:
          resume_df = self.df[self.df["ID"].astype(str) == id].iloc[0][["ID", "Resume"]]
          resume_with_id = "Applicant ID " + resume_df["ID"].astype(str) + "\n" + resume_df["Resume"]
          retrieved_resumes.append(resume_with_id)
        except IndexError:
          return []
      return retrieved_resumes

    @tool(args_schema=JobDescription)
    def retrieve_applicant_jd(job_description: str):
      """Retrieve similar resumes given a job description"""
      subquestion_list = [job_description]

      if rag_mode == "RAG Fusion":
        subquestion_list += llm.generate_subquestions(question)
        
      self.meta_data["subquestion_list"] = subquestion_list
      retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
      self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
      retrieved_resumes = self.retrieve_documents_with_id(retrieved_ids)
      return retrieved_resumes
    
    self.meta_data["rag_mode"] = rag_mode
    tools = [retrieve_applicant_id, retrieve_applicant_jd]
    normalized_question = question.lower()
    retrieval_hint = any(keyword in normalized_question for keyword in (
      "find", "search", "looking for", "hire", "need a", "candidate",
      "applicant", "resume", "python", "java", "developer", "developper",
      "engineer", "experience", "experince", "skill", "qualification",
      "requirement", "job description"
    ))
    if llm.provider in {"Hugging Face", "Groq"} or retrieval_hint and llm.provider == "Ollama":
      applicant_ids = re.findall(
        r"\b(?:applicant|candidate)\s+(?:id\s*)?#?([A-Za-z0-9_-]+)\b|\b(?:resume|id)\s*#([A-Za-z0-9_-]+)\b",
        question,
        re.IGNORECASE
      )
      applicant_ids = [first or second for first, second in applicant_ids]
      if applicant_ids:
        self.meta_data["query_type"] = "retrieve_applicant_id"
        self.meta_data["extracted_input"] = {"id_list": applicant_ids}
        return retrieve_applicant_id.invoke({"id_list": applicant_ids})
      self.meta_data["query_type"] = "retrieve_applicant_jd"
      self.meta_data["extracted_input"] = {"job_description": question}
      return retrieve_applicant_jd.invoke({"job_description": question})

    response = llm.llm.bind_tools(tools).invoke([
      ("system", "You are an expert in talent acquisition. Choose exactly one retrieval tool when the user asks about resumes or a job description. Otherwise, answer without using a tool."),
      ("user", question)
    ])

    retrieval_hint = retrieval_hint or len(question) >= 120
    if not response.tool_calls and retrieval_hint:
      response = llm.llm.bind_tools(tools, tool_choice="any").invoke([
        ("system", "Use the most appropriate retrieval tool for this resume or job-description request."),
        ("user", question)
      ])

    if not response.tool_calls:
      self.meta_data["query_type"] = "no_retrieve"
      return []

    tool_call = response.tool_calls[0]
    toolbox = {
      "retrieve_applicant_id": retrieve_applicant_id,
      "retrieve_applicant_jd": retrieve_applicant_jd
    }
    if tool_call["name"] not in toolbox:
      raise ValueError(f"Unsupported retrieval tool returned by model: {tool_call['name']}")

    self.meta_data["query_type"] = tool_call["name"]
    self.meta_data["extracted_input"] = tool_call["args"]
    return toolbox[tool_call["name"]].invoke(tool_call["args"])
