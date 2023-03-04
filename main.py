import pandas as pd
from Adaboost import AdaBoost
from bs4 import BeautifulSoup
import urllib.parse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

import ssl
import http.client
import warnings
warnings.filterwarnings('ignore')
import whois
import socket
from datetime import datetime,timedelta, timezone
import math
import numpy as np
from pyquery import PyQuery
from requests import get
import requests
import sys, getopt
from sklearn.linear_model import LinearRegression
from vecstack import stacking
from sklearn.ensemble import StackingClassifier
from urllib.parse import urlparse



class UrlFeaturizer:
    def __init__(self, url):
        self.url = url
        self.domain = url.split('//')[-1].split('/')[0]
        self.today = datetime.now().replace(tzinfo=None)


        try:
            self.response = get(self.url)
            self.pq = PyQuery(self.response.text)
        except:
            self.response = None
            self.pq = None
    def having_IP_Address(self):
        dataurl=[url]
        df=pd.DataFrame(dataurl,columns=['Col1'])
        a=df['Col1'].str.match('^(http|https)://\d+\.\d+\.\d+\.\d+\.*')
        if a[0]:
            return -1
        else:
            return 1
    def URL_Length(self):
        l=len(self.url)
        if l<54:
            return 1
        elif l>=54 and l<=75:
            return 0
        else:
            return -1
    def Shortining_Service(self):
        url=self.url
        response = requests.head(url, allow_redirects=True)
        if response.url != url:
            return -1
        else:
            return 1

    def atinurl(self):
        y = self.url
        if '@' in y:
            return -1
        else:
            return 1

    def last_double_slash_position_greater_than_7(self):
        url=self.url
        last_slash_pos = url.rfind('//')
        if last_slash_pos > 7:
            return -1
        else:
            return 1

    def domain_contains_hyphen(self):
        url=self.url
        domain = urllib.parse.urlparse(url).hostname
        if "-" in domain:
            return -1
        else:
            return 1
    def dots_in_url(self):
        url=self.url
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        num_dots = domain.count(".")
        if num_dots==1:
            return 1
        elif num_dots==2:
            return 0
        else:
            return -1

    def is_secure_url(self):
        url=self.url
        # Check if URL uses HTTPS
        if not url.startswith("https"):
            return -1

            ctx = ssl.create_default_context()
            with socket.create_connection((url.split("//")[1].split("/")[0], 443)) as sock:
                with ctx.wrap_socket(sock, server_hostname=url.split("//")[1].split("/")[0]) as ssock:
                    cert_pem = ssl.DER_cert_to_PEM_cert(ssock.getpeercert(binary_form=True))

            # Parse SSL certificate
            cert = x509.load_pem_x509_certificate(cert_pem.encode(), default_backend())
            not_after = cert.not_valid_after
            issuer = cert.issuer.rfc4514_string()
            # Check if certificate issuer is trusted
            if "Let's Encrypt" not in issuer and "DigiCert" not in issuer:
                return 0
        # Check if certificate is older than 1 year
            if not_after <= datetime.datetime.now() + datetime.timedelta(days=365):
                return 0
        return 1

    def is_domain_of_url_expiring_soon(self):
        # Extract domain name from URL
        url=self.url
        domain = urlparse(url).netloc

        # Retrieve domain registration information
        w = whois.whois(domain)

        # Extract expiration date
        expiration_date = w.expiration_date
        if type(expiration_date) == list:
            expiration_date = expiration_date[0]

        # Check if domain expires in less than 1 year
        if expiration_date <= datetime.now() + timedelta(days=365):
            return -1
        else:
            return 1
    def favicon(self):
        url=self.url
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        favicon_url = None
        for link in soup.find_all("link"):
            if "icon" in link.get("rel", []):
                favicon_url = link.get("href")
                break

        if favicon_url is not None:
            parsed_url = urlparse(favicon_url)
            if parsed_url.netloc != "" and parsed_url.netloc != urlparse(url).netloc:
                return -1
            else:
                return 1
        else:
            return 1

    def run(self):
        sample=[]
        sample.append(self.having_IP_Address())
        sample.append(self.URL_Length())
        sample.append(self.Shortining_Service())
        sample.append(self.atinurl())
        sample.append(self.last_double_slash_position_greater_than_7())
        sample.append(self.domain_contains_hyphen())
        sample.append(self.dots_in_url())
        sample.append(self.is_secure_url())
        sample.append(self.is_domain_of_url_expiring_soon())
        sample.append((self.favicon()))
        print(sample)
        return(sample)



df=pd.read_csv("C:\\Users\\HP\\data.csv")
x=df.iloc[:,0:10]
y=df.iloc[:,30]

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2)
ab=AdaBoostClassifier()
et=ExtraTreesClassifier()
#AdaModel=AdaBoostClassifier(n_estimators=100,learning_rate=1)
#extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,criterion ='entropy', max_features = 2)
base_models = [
    ('Adaboost', ab)
    ]

#model_1=AdaModel.fit(x_train,y_train)
#model_2=extra_tree_forest.fit(x_train,y_train)
stacked = StackingClassifier(estimators = base_models,final_estimator = et)



url=input()
#order=['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report']
a = UrlFeaturizer(url).run()


input=[a]
model=stacked.fit(x_train, y_train)
stacked_prediction = stacked.predict(input)
output = cross_val_score(stacked, x,y)
print("Accuracy",max(output))
print(stacked_prediction)