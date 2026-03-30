from langchain.agents import create_agent
from langchain.tools import tool
from models.logistic_regression import predict
from dotenv import load_dotenv

model = "meta-llama/llama-4-scout-17b-16e-instruct"

load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.tools import tool
import pandas as pd

class LogisticRegressionModel(BaseModel):
    int_rate: float = Field(description="Interest rate on the loan")
    installment: float = Field(description="The monthly payment owed if the loan originates")
    sub_grade: str = Field(description="Assigned loan subgrade")
    dti: float = Field(description="Debt-to-income ratio")
    initial_list_status: str = Field(description="Initial listing status of the loan")
    mths_since_last_delinq: float = Field(description="Months since the last delinquency")
    mths_since_last_record: float = Field(description="Months since the last public record")
    num_op_rev_tl: float = Field(description="Number of open revolving accounts")
    zip_code: str = Field(description="First 3 digits of the zip code")
    delinq_2yrs: float = Field(description="Number of 30+ days past due incidences in past 2 years")
    revol_util: float = Field(description="Revolving line utilization rate")
    term: str = Field(description="The number of payments on the loan (e.g., '36 months')")
    percent_bc_gt_75: float = Field(description="Percentage of all bankcard accounts > 75% limit")
    tot_hi_cred_lim: float = Field(description="Total high credit/limit")
    num_bc_sats: float = Field(description="Number of satisfactory bankcard accounts")
    num_il_tl: float = Field(description="Number of installment accounts")
    pct_tl_nvr_dlq: float = Field(description="Percent of trades never delinquent")
    tot_coll_amt: float = Field(description="Total collection amounts ever owed")
    emp_length: str = Field(description="Employment length in years")
    total_rev_hi_lim: float = Field(description="Total revolving high credit/limit")
    application_type: str = Field(description="Indicates whether the loan is an individual or joint app")
    open_acc: float = Field(description="Number of open credit lines")
    inq_last_6mths: float = Field(description="Number of inquiries in last 6 months")
    loan_amnt: float = Field(description="The listed amount of the loan")
    num_rev_accts: float = Field(description="Number of revolving accounts")
    tax_liens: float = Field(description="Number of tax liens")
    earliest_cr_line: str = Field(description="The month the earliest reported credit line was opened")
    revol_bal: float = Field(description="Total credit revolving balance")
    acc_now_delinq: float = Field(description="Number of accounts on which the borrower is now delinquent")
    num_bc_tl: float = Field(description="Number of bankcard accounts")
    fico_range_high: float = Field(description="FICO range high at time of application")
    addr_state: str = Field(description="The state provided by the borrower")
    home_ownership: str = Field(description="Home ownership status")
    tot_cur_bal: float = Field(description="Total current balance of all accounts")
    annual_inc: float = Field(description="The self-reported annual income")
    pub_rec_bankruptcies: float = Field(description="Number of public record bankruptcies")
    verification_status: str = Field(description="Indicates if income was verified")
    num_rev_tl_bal_gt_0: float = Field(description="Number of revolving trades with balance > 0")
    purpose: str = Field(description="A category provided by the borrower for the loan request")
    fico_range_low: float = Field(description="FICO range low at time of application")
    disbursement_method: str = Field(description="The method by which the borrower receives funds")
    collections_12_mths_ex_med: float = Field(description="Number of collections in 12 months excluding medical")
    total_acc: float = Field(description="The total number of credit lines currently in the borrower's file")
    pub_rec: float = Field(description="Number of derogatory public records")

@tool('get_prediction', args_schema=LogisticRegressionModel, return_direct=False)
def get_prediction(**kwargs) -> dict:
    """
    Predicts loan default. p_value 0 means the applicant will definitely pay off the loan and 1 means the applicant will definitely not pay the loan.
    RETURNS: A dictionary containing:
      - 'p_value': The probability of default (0.0 to 1.0)
      - 'threshold': The cutoff for approval
      - 'error': A string describing any input issues, or None if successful.
    """
    input_df = pd.DataFrame([kwargs])
    p, threshold, error = predict(input_df)
    return {
        "p_value": p,
        "threshold": threshold
    }

def main():



    agent = create_agent(
        model = f"groq:{model}",
        tools = [get_prediction],
        system_prompt = (
            'You are a helpful person and also cracks a lot of jokes (not random jokes but according to the guy) who helps people with their loan queries like whether a person will default on a specific loan given the person\'s details and whether my loan will be approved given my details'
            "When calling the 'get_prediction' tool, "
            "ensure that all numeric fields (like loan_amnt, int_rate, etc.) are passed "
            "as raw numbers, NOT as strings in quotes. For example: 'loan_amnt': 5000, not '5000'."
        )
    )
    prompt = input()
    response = agent.invoke({
        'messages': [
            {
                'role': 'user', 'content': prompt
            }
        ]
    })
    print(response)
    print(response['messages'][-1].content)

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