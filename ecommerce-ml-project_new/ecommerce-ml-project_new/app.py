from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('lr_model_new2.pkl')
label_encoder_product_category = joblib.load('label_encoder_product_category.pkl')
label_encoder_product = joblib.load('label_encoder_product_type.pkl')

# Mapping dictionaries for displaying human-readable labels
product_category_mapping = {0: 'Auto & Accessories', 1: 'Electronics', 2: 'Fashion', 3: 'Home & Furniture'}
product_mapping = {
    0: 'Apple Laptop', 1: 'Bed Sheets', 2: 'Beds', 3: 'Bike Tyres', 4: 'Car & Bike Care', 5: 'Car Body Covers',
    6: 'Car Mat', 7: 'Car Media Players', 8: 'Car Pillow & Neck Rest', 9: 'Car Seat Covers', 10: 'Car Speakers',
    11: 'Casual Shoes', 12: 'Curtains', 13: 'Dinner Crockery', 14: 'Dining Tables', 15: 'Fans', 16: 'Formal Shoes',
    17: 'Fossil Watch', 18: 'Iron', 19: 'Jeans', 20: 'Keyboard', 21: 'LCD', 22: 'LED', 23: 'Mixer/Juicer', 24: 'Mouse',
    25: 'Running Shoes', 26: 'Samsung Mobile', 27: 'Shirts', 28: 'Shoe Rack', 29: 'Sneakers', 30: 'Sofa Covers',
    31: 'Sofas', 32: 'Speakers', 33: 'Sports Wear', 34: 'Suits', 35: 'T-Shirts', 36: 'Tablet', 37: 'Titan Watch',
    38: 'Towels', 39: 'Tyre', 40: 'Umbrellas', 41: 'Watch'
}

# This route serves the HTML file
@app.route('/')
def home():
    return render_template('index.html', product_category_mapping=product_category_mapping, product_mapping=product_mapping, prediction_text=None)

# This route handles the prediction and also passes the mapping dictionaries
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        quantity = float(request.form['quantity'])
        discount = float(request.form['discount'])
        profit = float(request.form['profit'])
        shipping_cost = float(request.form['shipping_cost'])

        # Map categorical values to numerical using label encoders
        product_category = request.form['productCategory']
        product = request.form['product']

        # Update label encoders if new categories are encountered
        if product_category not in label_encoder_product_category.classes_:
            label_encoder_product_category.classes_ = numpy.append(label_encoder_product_category.classes_, product_category)

        if product not in label_encoder_product.classes_:
            label_encoder_product.classes_ = numpy.append(label_encoder_product.classes_, product)

        product_category_encoded = label_encoder_product_category.transform([product_category])[0]
        product_encoded = label_encoder_product.transform([product])[0]

        # Make a prediction using the pre-trained model
        prediction = model.predict([[quantity, discount, profit, shipping_cost, product_category_encoded, product_encoded]])[0]

        # Convert numerical values back to human-readable labels with default values
        product_category_label = product_category_mapping.get(product_category_encoded, 'Default Product Category')
        product_label = product_mapping.get(product_encoded, 'Default Product')

        # Format the prediction text
        prediction_text = f'Predicted Sales: ${round(prediction, 2)}'

        # Pass the mapping dictionaries to the template
        return render_template('index.html', product_category_mapping=product_category_mapping, product_mapping=product_mapping, prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', product_category_mapping=product_category_mapping, product_mapping=product_mapping, prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, port=5003)

    
    
    
    
    
    
