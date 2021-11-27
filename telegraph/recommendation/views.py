from django.shortcuts import render

# Create your views here.
from .forms import PairingForm
from .machinelearning import MachineLearning


def home(request):
    context = {
        'hello': 'hello world',
        'form': PairingForm
    }
    return render(request, 'recommendation/home.html', context)


def pair_wine(request):
    if request.method == 'POST':
        form = PairingForm(request.POST)
        if form.is_valid():
            food = form.cleaned_data['test_food']
            appetizer = form.cleaned_data['appetizer_list']
            entree = form.cleaned_data['entree_list']
            dessert = form.cleaned_data['dessert_list']
            food = MachineLearning.machine_learning(food, appetizer, entree, dessert)
            context = {
                'food': food,
                'appetizer': appetizer,
                'entree': entree,
                'dessert': dessert
            }

            return render(request, 'recommendation/home.html', context)

    else:
        form = PairingForm()

    return render(request, 'recommendation/home.html', {'form': form})
