from langchain.agents import create_agent
from langchain.tools import tool
from models.logistic_regression import predict
from dotenv import load_dotenv
from config import logging_config

model = "meta-llama/llama-4-scout-17b-16e-instruct"

load_dotenv()

logger = logging_config.get_logger(__name__)

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Optional, Union
import pandas as pd


# class LogisticRegressionModel(BaseModel):
#     int_rate: float = Field(description="Interest rate on the loan")
#     installment: float = Field(description="The monthly payment owed if the loan originates")
#     sub_grade: str = Field(description="Assigned loan subgrade")
#     dti: float = Field(description="Debt-to-income ratio")
#     initial_list_status: str = Field(description="Initial listing status of the loan")
#     mths_since_last_delinq: float = Field(description="Months since the last delinquency")
#     mths_since_last_record: float = Field(description="Months since the last public record")
#     num_op_rev_tl: float = Field(description="Number of open revolving accounts")
#     zip_code: str = Field(description="First 3 digits of the zip code")
#     delinq_2yrs: float = Field(description="Number of 30+ days past due incidences in past 2 years")
#     revol_util: float = Field(description="Revolving line utilization rate")
#     term: str = Field(description="The number of payments on the loan (e.g., '36 months')")
#     percent_bc_gt_75: float = Field(description="Percentage of all bankcard accounts > 75% limit")
#     tot_hi_cred_lim: float = Field(description="Total high credit/limit")
#     num_bc_sats: float = Field(description="Number of satisfactory bankcard accounts")
#     num_il_tl: float = Field(description="Number of installment accounts")
#     pct_tl_nvr_dlq: float = Field(description="Percent of trades never delinquent")
#     tot_coll_amt: float = Field(description="Total collection amounts ever owed")
#     emp_length: str = Field(description="Employment length in years")
#     total_rev_hi_lim: float = Field(description="Total revolving high credit/limit")
#     application_type: str = Field(description="Indicates whether the loan is an individual or joint app")
#     open_acc: float = Field(description="Number of open credit lines")
#     inq_last_6mths: float = Field(description="Number of inquiries in last 6 months")
#     loan_amnt: float = Field(description="The listed amount of the loan")
#     num_rev_accts: float = Field(description="Number of revolving accounts")
#     tax_liens: float = Field(description="Number of tax liens")
#     earliest_cr_line: str = Field(description="The month the earliest reported credit line was opened")
#     revol_bal: float = Field(description="Total credit revolving balance")
#     acc_now_delinq: float = Field(description="Number of accounts on which the borrower is now delinquent")
#     num_bc_tl: float = Field(description="Number of bankcard accounts")
#     fico_range_high: float = Field(description="FICO range high at time of application")
#     addr_state: str = Field(description="The state provided by the borrower")
#     home_ownership: str = Field(description="Home ownership status")
#     tot_cur_bal: float = Field(description="Total current balance of all accounts")
#     annual_inc: float = Field(description="The self-reported annual income")
#     pub_rec_bankruptcies: float = Field(description="Number of public record bankruptcies")
#     verification_status: str = Field(description="Indicates if income was verified")
#     num_rev_tl_bal_gt_0: float = Field(description="Number of revolving trades with balance > 0")
#     purpose: str = Field(description="A category provided by the borrower for the loan request")
#     fico_range_low: float = Field(description="FICO range low at time of application")
#     disbursement_method: str = Field(description="The method by which the borrower receives funds")
#     collections_12_mths_ex_med: float = Field(description="Number of collections in 12 months excluding medical")
#     total_acc: float = Field(description="The total number of credit lines currently in the borrower's file")
#     pub_rec: float = Field(description="Number of derogatory public records")

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
    logger.debug(f"llm get_prediction arguments: {kwargs}")
    input_df = pd.DataFrame([kwargs])
    p, threshold, error = predict(input_df)
    return {
        "p_value": p,
        "threshold": threshold
    }

def main(prompt: str):



    agent = create_agent(
        model = f"groq:{model}",
        # model_kwargs={"temperature": 0},
        tools = [get_prediction],
        system_prompt = (
            "You are a helpful Senior Loan Officer. "
            "Your goal is to use the 'get_prediction' tool as soon as you have the basic info. "
            "Do not ask the user to confirm information they already provided. Do not ask yes/no confirmation questions. Infer missing details from context and call get_prediction immediately."
            "\n\nRULES FOR TOOL CALLING:\n"
            "1. Only ask for: Loan Amount, Annual Income, and Term.\n"
            "2. If the user mentions home status (like 'renting'), use it. If not, don't ask.\n"
            "3. For ALL OTHER credit markers (FICO, DTI, etc.), do NOT ask the user. "
            "The tool will automatically use internal defaults. Call the tool immediately "
            "once you have the 3 core pieces of info.\n"
            "4. Be authoritative. The tool is Ground Truth."
        )
    )
    # prompt = input("please input your query: ")
    response = agent.invoke({
        'messages': [
            {
                'role': 'user', 'content': prompt
            }
        ]
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
    main()