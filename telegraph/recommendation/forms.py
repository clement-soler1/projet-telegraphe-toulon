from django import forms

test_food_list = [
    ('hotdog', 'Hotdog'),
    ('mustard', 'Mustard'),
    ('tomato', 'Tomato'),
    ('onion', 'Onion'),
    ('pepperoncini', 'Pepperoncini'),
    ('gherkin', 'Gherkin'),
    ('celery', 'Celery'),
    ('relish', 'Relish')
]

appetizer = [
    ('trout', 'Trout'),
    ('dill', 'Dill'),
    ('cucumber', 'Cucumber'),
    ('sour_cream', 'Sour Cream')
]

entree = [
    ('roast_chicken', 'Roast Chicken'),
    ('tarragon', 'Tarragon'),
    ('sage', 'Sage')
]

dessert = [
    ('peach', 'Peach'),
    ('pie', 'Pie')
]


class PairingForm(forms.Form):
    test_food = forms.MultipleChoiceField(label='Test food', choices=test_food_list, widget=forms.CheckboxSelectMultiple)
    appetizer_list = forms.MultipleChoiceField(label='Appetizer', choices=appetizer, widget=forms.CheckboxSelectMultiple)
    entree_list = forms.MultipleChoiceField(label='Entree', choices=entree, widget=forms.CheckboxSelectMultiple)
    dessert_list = forms.MultipleChoiceField(label='Dessert', choices=dessert, widget=forms.CheckboxSelectMultiple)
