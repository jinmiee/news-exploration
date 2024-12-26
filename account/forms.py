from django import forms
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import User
from . models import Feedbacks
import re

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Repeat Password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'email')

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError('Passwords don\'t match.')
        return cd['password2']

class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedbacks
        fields = ('rating', 'feedback')

class CustomPasswordChangeForm(PasswordChangeForm):
    def clean_new_password1(self):
        new_password1 = self.cleaned_data.get('new_password1')
        # 비밀번호 조건 확인
        if len(new_password1) < 8:
            raise forms.ValidationError("비밀번호는 최소 8자 이상이어야 합니다.")
        if not re.search(r'[A-Za-z]', new_password1):
            raise forms.ValidationError("비밀번호에 영문자를 포함해야 합니다.")
        if not re.search(r'[0-9]', new_password1):
            raise forms.ValidationError("비밀번호에 숫자를 포함해야 합니다.")
        if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', new_password1):
            raise forms.ValidationError("비밀번호에 특수문자를 포함해야 합니다.")
        return new_password1
