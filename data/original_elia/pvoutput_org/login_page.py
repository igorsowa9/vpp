import requests
from lxml import html
from bs4 import BeautifulSoup
import sys
from pprint import pprint as pp


s = requests.session()
login_url = "https://pvoutput.org"

response = s.get(login_url)
soup = BeautifulSoup(response.text, "html.parser")
pp(soup('input'))

payload = {
    'login': 'igorsowa9',
    'password': 'Zurawski9',
    'remember': '1'
}
s.post(login_url, data=payload, headers=dict(referer=login_url))
url = 'https://pvoutput.org/intraday.jsp?id=33196&sid=30411&dt=20180629'
result = s.get(url, headers=dict(referer=url))