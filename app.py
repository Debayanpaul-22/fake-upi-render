import pickle
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import *

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to UPI Detection System"

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
    
        data = request.get_json()
        app.logger.debug(f"Received JSON: {data}")
    
        Transaction_Type = str(data.get('Transaction_Type'))
        Payment_Gateway = str(data.get('Payment_Gateway'))
        Transaction_City = str(data.get('Transaction_City'))
        Transaction_State = str(data.get('Transaction_State'))
        Transaction_Status = str(data.get('Transaction_Status'))
        Device_OS = str(data.get('Device_OS'))
        Transaction_Frequency = int(data.get('Transaction_Frequency'))
        Merchant_Category = str(data.get('Merchant_Category'))
        Transaction_Channel = str(data.get('Transaction_Channel'))
        Transaction_Amount_Deviation =float(data.get('Transaction_Amount_Deviation'))
        Days_Since_Last_Transaction =int( data.get('Days_Since_Last_Transaction'))
        amount = float(data.get('amount'))

        input_quary = np.array([[Transaction_Type, Payment_Gateway, Transaction_City, Transaction_State, Transaction_Status
                                , Device_OS, Transaction_Frequency, Merchant_Category, Transaction_Channel
                                , Transaction_Amount_Deviation, Days_Since_Last_Transaction, amount]])

    
        new_df = pd.DataFrame(input_quary,columns=['Transaction_Type','Payment_Gateway','Transaction_City','Transaction_State','Transaction_Status'
                                               ,'Device_OS','Transaction_Frequency','Merchant_Category','Transaction_Channel','Transaction_Amount_Deviation'
                                               ,'Days_Since_Last_Transaction','amount'])
    
        le = LabelEncoder()
        new_df['Transaction_Type'] = le.fit_transform(new_df['Transaction_Type'])
        new_df['Payment_Gateway'] = le.fit_transform(new_df['Payment_Gateway'])
        new_df['Transaction_City'] = le.fit_transform(new_df['Transaction_City'])
        new_df['Transaction_State'] = le.fit_transform(new_df['Transaction_State'])
        new_df['Transaction_Status'] = le.fit_transform(new_df['Transaction_Status'])
        new_df['Device_OS'] = le.fit_transform(new_df['Device_OS'])
        new_df['Merchant_Category'] = le.fit_transform(new_df['Merchant_Category'])
        new_df['Transaction_Channel'] = le.fit_transform(new_df['Transaction_Channel'])
    
        result = model.predict(new_df)[0] 
    
        print("Prediction:",result)

        return jsonify({'fraud': str(result)})
    
    except Exception as e:
        app.logger.error(f"Exception in /predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    

