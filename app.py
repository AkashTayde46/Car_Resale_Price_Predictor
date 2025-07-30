
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import pickle
import numpy as np
from datetime import datetime
import random

app = Flask(__name__)

# Load the trained model and column names
model = joblib.load('final_model.pkl')
col_names = joblib.load('column_names.pkl')  # Ensure this is a list of column names

# Define encoding mappings based on the data analysis
FUEL_ENCODING = {
    'Petrol': 1,
    'Diesel': 2,
    'CNG': 3,
    'LPG': 4,
    'Electric': 5
}

INSURANCE_ENCODING = {
    'Third Party insurance': 1,
    'Comprehensive': 2,
    'Zero Dep': 3,
    'Third Party': 4,
    'Not Available': 5
}

OWNER_ENCODING = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth Owner': 4,
    'Fifth Owner': 5
}

TRANSMISSION_ENCODING = {
    'Manual': 1,
    'Automatic': 2
}

# Market analysis data (simulated for demonstration)
MARKET_TRENDS = {
    'price_trend': '+12%',
    'demand_level': 'High',
    'avg_sale_time': '45 days',
    'market_average': '₹5.2L',
    'prediction_accuracy': '98%',
    'cars_analyzed': '50K+'
}

def get_car_specific_features(car_name, fuel_type, year, transmission_type, seats):
    """Generate car-specific features and characteristics"""
    
    # Extract brand and model from car name
    car_parts = car_name.split()
    brand = car_parts[0] if car_parts else "Unknown"
    model_name = car_parts[1] if len(car_parts) > 1 else "Unknown"
    
    # Brand-specific features
    brand_features = {
        'Maruti': {
            'reliability': 'Excellent',
            'service_network': 'Extensive',
            'spare_parts': 'Readily Available',
            'fuel_efficiency': 'High',
            'resale_value': 'Strong',
            'brand_reputation': 'Trusted',
            'color_options': 'Wide Range'
        },
        'Hyundai': {
            'reliability': 'Very Good',
            'service_network': 'Wide',
            'spare_parts': 'Easily Available',
            'fuel_efficiency': 'Good',
            'resale_value': 'Good',
            'brand_reputation': 'Premium',
            'color_options': 'Modern Colors'
        },
        'Tata': {
            'reliability': 'Good',
            'service_network': 'Growing',
            'spare_parts': 'Available',
            'fuel_efficiency': 'Moderate',
            'resale_value': 'Stable',
            'brand_reputation': 'Indian Brand',
            'color_options': 'Standard Range'
        },
        'Honda': {
            'reliability': 'Excellent',
            'service_network': 'Premium',
            'spare_parts': 'Quality Parts',
            'fuel_efficiency': 'Very Good',
            'resale_value': 'Strong',
            'brand_reputation': 'Premium',
            'color_options': 'Elegant Options'
        },
        'Toyota': {
            'reliability': 'Outstanding',
            'service_network': 'Extensive',
            'spare_parts': 'Premium Quality',
            'fuel_efficiency': 'Excellent',
            'resale_value': 'Very Strong',
            'brand_reputation': 'Premium',
            'color_options': 'Sophisticated'
        },
        'Mahindra': {
            'reliability': 'Good',
            'service_network': 'Rural Focus',
            'spare_parts': 'Rugged',
            'fuel_efficiency': 'Moderate',
            'resale_value': 'Stable',
            'brand_reputation': 'SUV Specialist',
            'color_options': 'Adventure Colors'
        }
    }
    
    # Get brand features or use default
    brand_specific = brand_features.get(brand, {
        'reliability': 'Good',
        'service_network': 'Standard',
        'spare_parts': 'Available',
        'fuel_efficiency': 'Moderate',
        'resale_value': 'Stable',
        'brand_reputation': 'Reliable',
        'color_options': 'Standard'
    })
    
    # Fuel type specific features
    fuel_features = {
        'Petrol': {
            'engine_type': 'Petrol Engine',
            'power_output': 'Smooth Power Delivery',
            'maintenance': 'Lower Maintenance',
            'fuel_cost': 'Moderate',
            'environmental': 'Standard Emissions',
            'performance': 'Good Acceleration',
            'range': 'Standard Range'
        },
        'Diesel': {
            'engine_type': 'Diesel Engine',
            'power_output': 'High Torque',
            'maintenance': 'Higher Maintenance',
            'fuel_cost': 'Lower Cost',
            'environmental': 'Higher Emissions',
            'performance': 'Excellent Mileage',
            'range': 'Long Range'
        },
        'Electric': {
            'engine_type': 'Electric Motor',
            'power_output': 'Instant Torque',
            'maintenance': 'Minimal Maintenance',
            'fuel_cost': 'Very Low',
            'environmental': 'Zero Emissions',
            'performance': 'Silent Operation',
            'range': 'Limited Range'
        },
        'CNG': {
            'engine_type': 'CNG Compatible',
            'power_output': 'Eco-friendly',
            'maintenance': 'Low Maintenance',
            'fuel_cost': 'Very Low',
            'environmental': 'Clean Fuel',
            'performance': 'Economical',
            'range': 'Limited Range'
        },
        'LPG': {
            'engine_type': 'LPG Compatible',
            'power_output': 'Cost Effective',
            'maintenance': 'Low Maintenance',
            'fuel_cost': 'Low',
            'environmental': 'Cleaner than Petrol',
            'performance': 'Economical',
            'range': 'Standard Range'
        }
    }
    
    fuel_specific = fuel_features.get(fuel_type, fuel_features['Petrol'])
    
    # Year-based features
    year_features = {
        'safety_features': 'Basic Safety' if year < 2015 else 'Advanced Safety' if year >= 2020 else 'Standard Safety',
        'technology': 'Basic Features' if year < 2015 else 'Modern Tech' if year >= 2020 else 'Standard Tech',
        'emissions': 'BS4' if year >= 2017 else 'BS3' if year >= 2010 else 'Older Standards',
        'comfort': 'Basic Comfort' if year < 2015 else 'Premium Comfort' if year >= 2020 else 'Standard Comfort',
        'design': 'Classic Design' if year < 2015 else 'Modern Design' if year >= 2020 else 'Contemporary Design'
    }
    
    # Transmission features
    transmission_features = {
        'Manual': {
            'driving_experience': 'Engaging',
            'fuel_efficiency': 'Better',
            'maintenance_cost': 'Lower',
            'resale_value': 'Good',
            'learning_curve': 'Requires Skill'
        },
        'Automatic': {
            'driving_experience': 'Convenient',
            'fuel_efficiency': 'Moderate',
            'maintenance_cost': 'Higher',
            'resale_value': 'Better',
            'learning_curve': 'Easy to Drive'
        }
    }
    
    trans_specific = transmission_features.get(transmission_type, transmission_features['Manual'])
    
    # Seats-based features
    seat_features = {
        'family_friendly': 'Yes' if seats >= 5 else 'Limited',
        'cargo_space': 'Good' if seats <= 5 else 'Limited',
        'comfort_level': 'Spacious' if seats <= 5 else 'Compact',
        'utility': 'Family Car' if seats >= 5 else 'City Car',
        'parking': 'Easy' if seats <= 5 else 'Challenging'
    }
    
    # Combine all features
    car_features = {
        'brand_info': {
            'brand': brand,
            'model': model_name,
            'full_name': car_name,
            'year': year
        },
        'brand_features': brand_specific,
        'fuel_features': fuel_specific,
        'year_features': year_features,
        'transmission_features': trans_specific,
        'seat_features': seat_features,
        'overall_rating': calculate_overall_rating(year, brand, fuel_type),
        'unique_selling_points': get_unique_selling_points(car_name, fuel_type, year, transmission_type, seats)
    }
    
    return car_features

def calculate_overall_rating(year, brand, fuel_type):
    """Calculate overall car rating based on specifications"""
    rating = 7.0  # Base rating
    
    # Year factor
    if year >= 2020:
        rating += 1.5
    elif year >= 2015:
        rating += 1.0
    elif year >= 2010:
        rating += 0.5
    
    # Brand factor
    premium_brands = ['Toyota', 'Honda', 'Hyundai']
    if brand in premium_brands:
        rating += 0.5
    
    # Fuel type factor
    if fuel_type == 'Electric':
        rating += 1.0
    elif fuel_type == 'Petrol':
        rating += 0.3
    elif fuel_type == 'Diesel':
        rating += 0.2
    
    return min(10.0, round(rating, 1))

def get_unique_selling_points(car_name, fuel_type, year, transmission_type, seats):
    """Generate unique selling points for the car"""
    points = []
    
    # Brand-specific points
    if 'Maruti' in car_name:
        points.append("Excellent fuel efficiency and low maintenance")
        points.append("Wide service network across India")
    elif 'Hyundai' in car_name:
        points.append("Premium features and modern design")
        points.append("Good resale value and reliability")
    elif 'Tata' in car_name:
        points.append("Strong build quality and safety features")
        points.append("Good value for money")
    elif 'Honda' in car_name:
        points.append("Premium engineering and smooth performance")
        points.append("Excellent reliability and resale value")
    elif 'Toyota' in car_name:
        points.append("Outstanding reliability and durability")
        points.append("Excellent resale value and low maintenance")
    
    # Fuel type points
    if fuel_type == 'Electric':
        points.append("Zero emissions and very low running cost")
        points.append("Future-ready technology and government incentives")
    elif fuel_type == 'Petrol':
        points.append("Smooth performance and lower maintenance")
        points.append("Good for city driving and short trips")
    elif fuel_type == 'Diesel':
        points.append("Excellent fuel efficiency for long drives")
        points.append("High torque and good for highway driving")
    
    # Year-based points
    if year >= 2020:
        points.append("Latest safety features and technology")
        points.append("Modern design and premium feel")
    elif year >= 2015:
        points.append("Good balance of features and value")
        points.append("Reliable performance and decent technology")
    
    # Transmission points
    if transmission_type == 'Automatic':
        points.append("Convenient automatic transmission")
        points.append("Easy to drive in city traffic")
    else:
        points.append("Better fuel efficiency with manual transmission")
        points.append("More engaging driving experience")
    
    return points[:6]  # Return top 6 points

def get_market_insights(car_name, fuel_type, year):
    """Generate market insights based on car specifications"""
    insights = {
        'market_position': 'Above Average' if year >= 2018 else 'Average',
        'depreciation_rate': f"{max(5, 20 - (2024 - year))}% per year",
        'demand_trend': 'Increasing' if fuel_type in ['Electric', 'Petrol'] else 'Stable',
        'best_selling_season': 'Monsoon' if fuel_type == 'Petrol' else 'Winter',
        'maintenance_cost': 'Low' if fuel_type == 'Electric' else 'Medium',
        'resale_value_retention': 'High' if year >= 2020 else 'Medium'
    }
    return insights

def get_price_comparison(predicted_price, car_name, fuel_type):
    """Generate price comparison with similar cars"""
    # Simulate price comparison data
    base_price = predicted_price
    comparisons = {
        'similar_cars': [
            {
                'name': f"{car_name.split()[0]} {car_name.split()[1]} (Similar Model)",
                'price': round(base_price * random.uniform(0.9, 1.1), 2),
                'difference': round((base_price * random.uniform(0.9, 1.1) - base_price), 2)
            },
            {
                'name': f"{car_name.split()[0]} {car_name.split()[1]} (Different Variant)",
                'price': round(base_price * random.uniform(0.85, 1.15), 2),
                'difference': round((base_price * random.uniform(0.85, 1.15) - base_price), 2)
            }
        ],
        'market_range': {
            'min': round(base_price * 0.8, 2),
            'max': round(base_price * 1.2, 2),
            'avg': round(base_price, 2)
        }
    }
    return comparisons

def get_selling_tips(car_name, fuel_type, year, predicted_price):
    """Generate personalized selling tips"""
    tips = []
    
    if year >= 2020:
        tips.append("Highlight the modern features and low mileage")
        tips.append("Emphasize the warranty period if applicable")
    elif year >= 2015:
        tips.append("Focus on the car's reliability and maintenance history")
        tips.append("Showcase any recent repairs or upgrades")
    else:
        tips.append("Emphasize the car's condition and service history")
        tips.append("Consider selling to a dealer for better value")
    
    if fuel_type == 'Electric':
        tips.append("Highlight the environmental benefits and fuel savings")
        tips.append("Mention the battery health and charging infrastructure")
    elif fuel_type == 'Petrol':
        tips.append("Emphasize the lower maintenance costs compared to diesel")
        tips.append("Highlight the smooth driving experience")
    elif fuel_type == 'Diesel':
        tips.append("Focus on the fuel efficiency and torque")
        tips.append("Mention the long-term cost benefits")
    
    if predicted_price > 8:
        tips.append("Consider professional photography for better presentation")
        tips.append("Target premium buyers who value quality")
    else:
        tips.append("Price competitively to attract more buyers")
        tips.append("Consider selling during peak buying seasons")
    
    return tips

@app.route('/')
def home_page():
    # Read the car_encoded.csv file to get car names for dropdown
    try:
        car_mapping_df = pd.read_csv('static/car_encoded.csv')
        car_names = car_mapping_df['car_name'].tolist()
    except Exception as e:
        car_names = []
        print(f"Error reading car_encoded.csv: {e}")
    
    return render_template('home.html', car_names=car_names)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract form data
        feat_data = request.form.to_dict()
        
        # Handle car name selection - convert to encoded value
        if 'car_name' in feat_data:
            car_name = feat_data['car_name']
            # Read the mapping to get encoded value
            car_mapping_df = pd.read_csv('static/car_encoded.csv')
            car_row = car_mapping_df[car_mapping_df['car_name'] == car_name]
            if not car_row.empty:
                encoded_value = car_row['Encoded_value'].iloc[0]
                feat_data['name_encoded'] = encoded_value
            else:
                return render_template('after.html', data="Error: Invalid car name selected")
        
        # Convert fuel type to encoded value
        if 'fuel_type' in feat_data:
            fuel_type = feat_data['fuel_type']
            if fuel_type in FUEL_ENCODING:
                feat_data['fuel'] = FUEL_ENCODING[fuel_type]
            else:
                return render_template('after.html', data="Error: Invalid fuel type selected")
        
        # Convert insurance type to encoded value
        if 'insurance_type' in feat_data:
            insurance_type = feat_data['insurance_type']
            if insurance_type in INSURANCE_ENCODING:
                feat_data['insurance_type'] = INSURANCE_ENCODING[insurance_type]
            else:
                return render_template('after.html', data="Error: Invalid insurance type selected")
        
        # Convert owner type to encoded value
        if 'owner_type' in feat_data:
            owner_type = feat_data['owner_type']
            if owner_type in OWNER_ENCODING:
                feat_data['Owner'] = OWNER_ENCODING[owner_type]
            else:
                return render_template('after.html', data="Error: Invalid owner type selected")
        
        # Convert transmission type to encoded value
        if 'transmission_type' in feat_data:
            transmission_type = feat_data['transmission_type']
            if transmission_type in TRANSMISSION_ENCODING:
                feat_data['transmission'] = TRANSMISSION_ENCODING[transmission_type]
            else:
                return render_template('after.html', data="Error: Invalid transmission type selected")
        
        # Remove original field names that are not needed for prediction
        fields_to_remove = ['car_name', 'fuel_type', 'owner_type', 'transmission_type']
        for field in fields_to_remove:
            if field in feat_data:
                del feat_data[field]
        
        # Validate numeric inputs
        try:
            feat_data['kms_driven'] = int(feat_data['kms_driven'])
            feat_data['seats'] = int(feat_data['seats'])
            feat_data['year'] = int(feat_data['year'])
        except ValueError:
            return render_template('after.html', data="Error: Invalid numeric input")
        
        # Validate ranges
        if not (1000 <= feat_data['kms_driven'] <= 500000):
            return render_template('after.html', data="Error: Kilometers driven must be between 1,000 and 500,000")
        if not (2 <= feat_data['seats'] <= 14):
            return render_template('after.html', data="Error: Number of seats must be between 2 and 14")
        if not (1990 <= feat_data['year'] <= 2024):
            return render_template('after.html', data="Error: Year must be between 1990 and 2024")
        
        # Create DataFrame
        df = pd.DataFrame([feat_data])
        df = df.reindex(columns=col_names)  # Ensure correct column order
        
        # Make prediction
        prediction = model.predict(df)
        predicted_price = float(prediction[0])
        
        # Generate additional insights
        original_car_name = request.form.get('car_name', '')
        original_fuel_type = request.form.get('fuel_type', '')
        original_year = int(request.form.get('year', 2020))
        original_transmission = request.form.get('transmission_type', '')
        original_seats = int(request.form.get('seats', 5))
        
        # Generate car-specific features
        car_features = get_car_specific_features(
            original_car_name, 
            original_fuel_type, 
            original_year, 
            original_transmission, 
            original_seats
        )
        
        market_insights = get_market_insights(original_car_name, original_fuel_type, original_year)
        price_comparison = get_price_comparison(predicted_price, original_car_name, original_fuel_type)
        selling_tips = get_selling_tips(original_car_name, original_fuel_type, original_year, predicted_price)
        
        # Render the result template with prediction and insights
        return render_template('after.html', 
                             data=predicted_price,
                             market_insights=market_insights,
                             price_comparison=price_comparison,
                             selling_tips=selling_tips,
                             car_features=car_features,
                             car_details={
                                 'name': original_car_name,
                                 'fuel_type': original_fuel_type,
                                 'year': original_year,
                                 'kms_driven': feat_data['kms_driven'],
                                 'seats': feat_data['seats']
                             })
    except Exception as e:
        # Handle errors gracefully
        return render_template('after.html', data=f"Error: {str(e)}")

@app.route('/api/market-trends')
def get_market_trends():
    """API endpoint for market trends"""
    return jsonify(MARKET_TRENDS)

@app.route('/api/car-suggestions')
def get_car_suggestions():
    """API endpoint for car suggestions based on budget"""
    try:
        car_mapping_df = pd.read_csv('static/car_encoded.csv')
        suggestions = car_mapping_df.sample(n=min(10, len(car_mapping_df))).to_dict('records')
        return jsonify(suggestions)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
