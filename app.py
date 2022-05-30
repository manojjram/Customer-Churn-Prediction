# save this as app.py
from flask import Flask, escape, request, render_template
import numpy as np
import pandas as pd
import xgboost as xgb

import pickle
model = pickle.load(open('xgboost_model.pkl', 'rb'))

# model = xgb.Booster()
# model.load_model("model.txt")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analysis')
def analysis():
    return render_template("profiling.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        cid=request.form['cid']
        name=request.form['name']
        age=int(request.form['age'])
        last_login=int(request.form['last_login'])
        avg_time_spent=float(request.form['avg_time_spent'])
        avg_transaction_value=float(request.form['avg_transaction_value'])
        points_in_wallet=float(request.form['points_in_wallet'])
        date=request.form['date']
        gender=request.form['gender']
        region_category_st=request.form['region_category']
        membership_category=request.form['membership_category']
        joined_through_referral=request.form['joined_through_referral']
        preferred_offer_types=request.form['preferred_offer_types']
        medium_of_operation=request.form['medium_of_operation']
        internet_option=request.form['internet_option']
        used_special_discount=request.form['used_special_discount']
        offer_application_preference=request.form['offer_application_preference']
        avg_frequency_login_days=float(request.form['avg_frequency_login_days'])
        complaint_status_st=request.form['complaint_status']
        past_complaint=request.form['past_complaint']
        feedback_st=request.form['feedback']

        # gender
        if gender=="M":
            gender = 1
        elif gender=="Unknown":
            gender=2
        else:
            gender=0
        
        # region_category
        if region_category_st == 'Town':
            region_category = 2
        if region_category_st == 'Village':
            region_category=0
        else:
            region_category=1

        # membership_category
        if membership_category=='Gold Membership':
            membership_category = 3
        elif membership_category=='No Membership':
            membership_category = 2

        elif membership_category=='Platinum Membership':
            membership_category = 0

        elif membership_category=='Silver Membership':
            membership_category = 4
        elif membership_category=='Premium Membership':
            membership_category = 1
        else:
            membership_category = 5

        # joined_through_referral
        if joined_through_referral=='Yes':
            joined_through_referral = 1
        else:
            joined_through_referral = 0

        # preferred_offer_types
        if preferred_offer_types=='Gift Vouchers/Coupons':
            preferred_offer_types=0
        if preferred_offer_types=='Without Offers':
            preferred_offer_types=2
        else:
            preferred_offer_types=1

        # medium_of_operation
        if medium_of_operation=='Desktop':
            medium_of_operation = 0

        elif medium_of_operation=='Smartphone':
            medium_of_operation = 1

        else:
            medium_of_operation = 2


        # internet_option
        if internet_option == 'Mobile_Data':
            internet_option = 1
        elif internet_option == 'Wi-Fi':
            internet_option = 0
        else:
            internet_option = 2


        # used_special_discount
        if used_special_discount=='Yes':
            used_special_discount=0
        else:
            used_special_discount=1

        # offer_application_preference
        if offer_application_preference=='Yes':
            offer_application_preference=0
        else:
            offer_application_preference=1

        # past_complaint
        if past_complaint=='Yes':
            past_complaint=1
        else:
            past_complaint=0

        # complaint_status
        if complaint_status_st=='Not Applicable':
            complaint_status=0
        elif complaint_status_st=='Solved':
            complaint_status=1
        elif complaint_status_st=='Solved in Follow-up':
            complaint_status=2
        elif complaint_status_st=='Unsolved':
            complaint_status=3
        else:
            complaint_status=4

        # feedback
        if feedback_st =='Poor Customer Service':
            feedback=5
        elif feedback_st =='Poor Product Quality':
            feedback=4
        elif feedback_st =='Poor Website':
            feedback=2
        elif feedback_st =='Products always in Stock':
            feedback=0
        elif feedback_st =='Quality Customer Care':
            feedback=1
        elif feedback_st =='Reasonable Price':
            feedback=8
        elif feedback_st =='Too many ads':
            feedback=6
        elif feedback_st =='User Friendly Website':
            feedback=7
        else:
            feedback=3
       


        joining_date = pd.to_datetime(date)
        days_since_joined = (pd.Timestamp('today') - joining_date).days

        data = {'age':[age], 'days_since_last_login':[last_login], 'avg_time_spent':[avg_time_spent], 'avg_transaction_value':[avg_transaction_value], 'points_in_wallet':[points_in_wallet],'days_since_joined':[days_since_joined],'gender':[gender], 'region_category':[region_category], 'membership_category':[membership_category], 'joined_through_referral':[joined_through_referral], 'preferred_offer_types':[preferred_offer_types], 'medium_of_operation':[medium_of_operation], 'internet_option':[internet_option], 'used_special_discount':[used_special_discount], 'offer_application_preference':[offer_application_preference], 'past_complaint':[past_complaint],'avg_frequency_login_days':[avg_frequency_login_days],'complaint_status':[complaint_status],'feedback':feedback}

        
        df = pd.DataFrame.from_dict(data)

        cols = model.get_booster().feature_names    # to avoid mismatch
        df = df[cols]

        prediction = float(model.predict(df))
        prediction_percent=(prediction/5)*100
        print(prediction_percent)
        return render_template("chart.html",cid=cid,name=name,age=age,gender=gender,membership_category=membership_category,avg_transaction_value=avg_transaction_value, prediction_percent=prediction_percent,feedback_st=feedback_st,complaint_status_st=complaint_status_st,points_in_wallet=points_in_wallet,region_category_st=region_category_st,prediction=prediction)        

    else:
        return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)
