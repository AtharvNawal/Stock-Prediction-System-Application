from django.contrib.auth.models import User
from django.test import TestCase


class AccountTests(TestCase):
    """Sign up, log in and log out. /history/ is used to check the header because it needs no market data."""

    def test_signup_creates_account_and_logs_in(self):
        response = self.client.post('/accounts/signup/', {
            'username': 'asha', 'email': 'asha@example.com',
            'password1': 'Tr1cky-passw0rd', 'password2': 'Tr1cky-passw0rd',
        })
        self.assertRedirects(response, '/', fetch_redirect_response=False)
        self.assertTrue(User.objects.filter(username='asha').exists())
        page = self.client.get('/history/').content.decode()
        self.assertIn('asha', page)
        self.assertIn('Logout', page)
        self.assertNotIn('Get free account', page)

    def test_signup_rejects_mismatched_passwords(self):
        response = self.client.post('/accounts/signup/', {
            'username': 'asha', 'password1': 'Tr1cky-passw0rd', 'password2': 'different-passw0rd',
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='asha').exists())
        self.assertIn('auth-field-error', response.content.decode())

    def test_login_and_logout(self):
        User.objects.create_user('ravi', password='Tr1cky-passw0rd')
        response = self.client.post('/accounts/login/', {'username': 'ravi', 'password': 'Tr1cky-passw0rd',
                                                          'next': '/history/'})
        self.assertRedirects(response, '/history/', fetch_redirect_response=False)
        self.assertIn('Logout', self.client.get('/history/').content.decode())

        response = self.client.post('/accounts/logout/')
        self.assertRedirects(response, '/', fetch_redirect_response=False)
        page = self.client.get('/history/').content.decode()
        self.assertIn('Login', page)
        self.assertIn('Get free account', page)

    def test_wrong_password_shows_error(self):
        User.objects.create_user('ravi', password='Tr1cky-passw0rd')
        response = self.client.post('/accounts/login/', {'username': 'ravi', 'password': 'wrong'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('auth-error', response.content.decode())
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_logged_out_header_shows_login_and_signup(self):
        page = self.client.get('/history/').content.decode()
        self.assertIn('/accounts/login/', page)
        self.assertIn('/accounts/signup/', page)
