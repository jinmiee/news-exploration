from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm
from .models import Article


# Create your views here.
# @login_required
def chart(request):
    data = Article.objects.all()[:5]
    context = {'section': 'chart', 'videos': data}
    return render(request, 'analysis/chart.html', context) 

# @login_required
def emotion(request):
    return render(request, 'analysis/emotion.html', {'section': 'emotion'}) 

# @login_required
def relate(request):
    return render(request, 'analysis/relate.html', {'section': 'relate'}) 

# @login_required
def mypage(request):
    return render(request, 'analysis/mypage.html', {'section': 'mypage'}) 


def register(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            # Create a new user object but avoid saving it yet
            new_user = user_form.save(commit=False)
            # Set the chosen password
            new_user.set_password(user_form.cleaned_data['password'])
            # Save the User object
            new_user.save()
            return render(request, 'registration/register_done.html', {'new_user': new_user})
    else:
        user_form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'user_form': user_form})