from langchain.agents import create_agent
from langchain.tools import tool
from models.logistic_regression import predict
from dotenv import load_dotenv
from config import logging_config
from machine_learning.serializers import ChatbotPredictData

model = "meta-llama/llama-4-scout-17b-16e-instruct"

load_dotenv()

logger = logging_config.get_logger(__name__)

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Optional, Union
import pandas as pd

class LogisticRegressionModel(BaseModel):
    # CORE USER FIELDS (LLM should ask for these)
    loan_amnt: Union[float, str] = Field(description="The requested loan amount")
    annual_inc: Union[float, str] = Field(description="The borrower's self-reported annual income")
    term: str = Field(description="The term of the loan in months (not years), e.g., '36 months' or '60 months'")
    emp_length: str = Field(default="10+ years", description="Employment length (e.g., '10+ years', '1 year')")
    home_ownership: str = Field(default="MORTGAGE", description="Home status (RENT, OWN, MORTGAGE)")
    purpose: str = Field(default="debt_consolidation", description="Purpose of the loan")
    addr_state: str = Field(default="CA", description="Borrower state (2-letter code)")
    verification_status: str = Field(default="Not Verified", description="Income verification status")

    # CREDIT MARKERS (Defaults provided so the user doesn't have to provide them)
    int_rate: Optional[Union[float, str]] = Field(default=12.0)
    installment: Optional[Union[float, str]] = Field(default=150.0)
    sub_grade: Optional[str] = Field(default="B1")
    dti: Optional[Union[float, str]] = Field(default=15.0)
    fico_range_low: Optional[Union[float, str]] = Field(default=700.0)
    fico_range_high: Optional[Union[float, str]] = Field(default=740.0)
    revol_bal: Optional[Union[float, str]] = Field(default=5000.0)
    revol_util: Optional[Union[float, str]] = Field(default=30.0)
    total_acc: Optional[Union[float, str]] = Field(default=15.0)
    open_acc: Optional[Union[float, str]] = Field(default=8.0)
    
    # DELINQUENCY & TECHNICAL MARKERS (Default to 'Clean' / 0.0)
    delinq_2yrs: Optional[Union[float, str]] = Field(default=0.0)
    pub_rec: Optional[Union[float, str]] = Field(default=0.0)
    pub_rec_bankruptcies: Optional[Union[float, str]] = Field(default=0.0)
    tax_liens: Optional[Union[float, str]] = Field(default=0.0)
    acc_now_delinq: Optional[Union[float, str]] = Field(default=0.0)
    tot_coll_amt: Optional[Union[float, str]] = Field(default=0.0)
    tot_cur_bal: Optional[Union[float, str]] = Field(default=10000.0)
    total_rev_hi_lim: Optional[Union[float, str]] = Field(default=10000.0)
    tot_hi_cred_lim: Optional[Union[float, str]] = Field(default=10000.0)
    collections_12_mths_ex_med: Optional[Union[float, str]] = Field(default=0.0)
    inq_last_6mths: Optional[Union[float, str]] = Field(default=0.0)
    mths_since_last_delinq: Optional[Union[float, str]] = Field(default=0.0)
    mths_since_last_record: Optional[Union[float, str]] = Field(default=0.0)
    num_op_rev_tl: Optional[Union[float, str]] = Field(default=5.0)
    percent_bc_gt_75: Optional[Union[float, str]] = Field(default=0.0)
    num_bc_sats: Optional[Union[float, str]] = Field(default=3.0)
    num_il_tl: Optional[Union[float, str]] = Field(default=5.0)
    pct_tl_nvr_dlq: Optional[Union[float, str]] = Field(default=100.0)
    num_rev_accts: Optional[Union[float, str]] = Field(default=10.0)
    num_bc_tl: Optional[Union[float, str]] = Field(default=5.0)
    num_rev_tl_bal_gt_0: Optional[Union[float, str]] = Field(default=2.0)
    
    # METADATA
    zip_code: Optional[str] = Field(default="902")
    initial_list_status: Optional[str] = Field(default="f")
    earliest_cr_line: Optional[str] = Field(default="Jan-2010")
    disbursement_method: Optional[str] = Field(default="Cash")
    application_type: Optional[str] = Field(default="Individual")

@tool('get_prediction', args_schema=LogisticRegressionModel, return_direct=False)
def get_prediction(**kwargs) -> dict:
    """
    Predicts loan default. p_value 0 means the applicant will definitely pay off the loan and 1 means the applicant will definitely not pay the loan.
    RETURNS: A dictionary containing:
      - 'p_value': The probability of default (0.0 to 1.0)
      - 'threshold': The cutoff for approval
      - 'error': A string describing any input issues, or None if successful.
    """
    logger.info("get_prediction tool is called.")
    input_df = pd.DataFrame([kwargs])
    p, threshold, error = predict(input_df)
    return {
        "p_value": p,
        "threshold": threshold
    }

def main(prompt: ChatbotPredictData):

    agent = create_agent(
        model = f"groq:{model}",
        # model_kwargs={"temperature": 0},
        tools = [get_prediction],
        system_prompt = (
            "You are a senior loan officer assistant. Your job is to run a default-risk assessment using the get_prediction tool."
            "Required from the user (ask only if missing):"
            "- Loan amount"
            "- Annual income"
            "- Loan term (in months, e.g. 36 or 60)"
            "Optional: If the user states home status (e.g. renting, owning, mortgage), pass it through; otherwise do not ask."
            "Do not ask for: FICO, DTI, interest rate, utilization, or any other credit markers—the tool supplies defaults for those. Do not ask yes/no questions to confirm details the user already gave. Reasonably infer unstated basics from context when possible."
            "As soon as loan amount, income, and term are known (or clearly implied), call get_prediction without delay. Treat the tool’s returned probability and threshold as the assessment you present; explain them clearly and confidently to the user."
        )
    )
    # prompt = input("please input your query: ")
    messages = [
        {
            'role': 'user', 'content': prompt.prompt
        }
    ]
    for turn in prompt.chat_history:
        messages.append({
            'role': 'user', 'content': turn.prompt
        })
        messages.append({
            'role': 'assistant', 'content': turn.response
        })
    response = agent.invoke({
        'messages': messages
    })
    # print(response)
    # print(response['messages'][-1].content)
    return response['messages'][-1].content

# def main2():
#     content = input()
#     client = Groq()
#     completion = client.chat.completions.create(
#         model="meta-llama/llama-4-scout-17b-16e-instruct",
#         messages=[
#         {
#             "role": "user",
#             "content": ""
#         }
#         ],
#         temperature=1,
#         max_completion_tokens=1024,
#         top_p=1,
#         stream=True,
#         stop=None
#     )

#     for chunk in completion:
#         print(chunk.choices[0].delta.content or "", end="")

# def main2():
#     # 2. Initialize Groq LLM
#     llm = ChatGroq(
#         model="meta-llama/llama-4-scout-17b-16e-instruct",
#         groq_api_key=api_key,
#         temperature=0
#     )

#     # 3. Pull the standard ReAct prompt template
#     # This template contains the "Thought/Action/Observation" logic the LLM needs
#     base_prompt = hub.pull("hwchase17/react")
    
#     # 4. Create the Agent
#     # We pass the system instructions by modifying the base prompt
#     # In this version, we use 'create_agent' which is the stable V1.0+ way
#     agent = create_agent(llm, [get_prediction], base_prompt)

#     # 5. Create the Executor (This actually runs the agent loop)
#     agent_executor = AgentExecutor(
#         agent=agent, 
#         tools=[get_prediction], 
#         verbose=True, 
#         handle_parsing_errors=True
#     )

#     # 6. Execution
#     print("System: I'm ready. I treat the prediction tool as Ground Truth.")
#     user_query = input("User: ")
    
#     # Standard invoke for the new AgentExecutor
#     response = agent_executor.invoke({"input": user_query})
    
#     print(f"\nAI: {response['output']}")

if __name__ == "__main__":
    main(input("Give Query"))