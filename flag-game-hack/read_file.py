import json
from fuzzywuzzy import fuzz

with open('countries.json') as f:
    countries = json.load(f)

def get_potential_country(recognized_country_name):
    max_score = 0
    potential_country = {}

    for country in countries:
        country_name = country['name']
        result = fuzz.ratio(recognized_country_name, country_name)
        if result > 50:
            if max_score < result:
                max_score = result
                potential_country = country
    return (max_score, potential_country)

country_data = get_potential_country('Бангладеж')[1]
print(country_data)