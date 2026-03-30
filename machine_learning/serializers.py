from pydantic_core.core_schema import DataclassSchema
from rest_framework import serializers

from rest_framework import serializers
from pydantic import BaseModel
class ChatbotPredictData(BaseModel):
    prompt: str

class LogisticRegressionDataSerializer(serializers.Serializer):
    int_rate = serializers.FloatField()
    installment = serializers.FloatField()
    sub_grade = serializers.CharField(max_length=10)
    dti = serializers.FloatField()
    initial_list_status = serializers.CharField(max_length=10)
    mths_since_last_delinq = serializers.FloatField()
    mths_since_last_record = serializers.FloatField()
    num_op_rev_tl = serializers.FloatField()
    zip_code = serializers.CharField(max_length=10)
    delinq_2yrs = serializers.FloatField()
    revol_util = serializers.FloatField()
    term = serializers.CharField(max_length=32)
    percent_bc_gt_75 = serializers.FloatField()
    tot_hi_cred_lim = serializers.FloatField()
    num_bc_sats = serializers.FloatField()
    num_il_tl = serializers.FloatField()
    pct_tl_nvr_dlq = serializers.FloatField()
    tot_coll_amt = serializers.FloatField()
    emp_length = serializers.CharField(max_length=32)
    total_rev_hi_lim = serializers.FloatField()
    application_type = serializers.CharField(max_length=32)
    open_acc = serializers.FloatField()
    inq_last_6mths = serializers.FloatField()
    loan_amnt = serializers.FloatField()
    num_rev_accts = serializers.FloatField()
    tax_liens = serializers.FloatField()
    earliest_cr_line = serializers.CharField(max_length=32)
    revol_bal = serializers.FloatField()
    acc_now_delinq = serializers.FloatField()
    num_bc_tl = serializers.FloatField()
    fico_range_high = serializers.FloatField()
    addr_state = serializers.CharField(max_length=8)
    home_ownership = serializers.CharField(max_length=16)
    tot_cur_bal = serializers.FloatField()
    annual_inc = serializers.FloatField()
    pub_rec_bankruptcies = serializers.FloatField()
    verification_status = serializers.CharField(max_length=32)
    num_rev_tl_bal_gt_0 = serializers.FloatField()
    purpose = serializers.CharField(max_length=64)
    fico_range_low = serializers.FloatField()
    disbursement_method = serializers.CharField(max_length=32)
    collections_12_mths_ex_med = serializers.FloatField()
    total_acc = serializers.FloatField()
    pub_rec = serializers.FloatField()


class LogisticRegressionPredictSerializer(serializers.Serializer):
    data = LogisticRegressionDataSerializer()