"""  DynaSwapApp/models.py  """
from django.db import models

# Create your models here.
class Roles(models.Model):
    """  Roles Class  """
    role_name = models.CharField(max_length=32, unique=True)
    url = models.URLField("URL")

    def __str__(self):
        return self.role_name

# INSERT INTO dynaswapapp_roles(role_name,url)
#   VALUES('President','/images/rs_1.jpg'),
#         ('CEO','/images/rs_2.jpg'),
#         ('Manager','/images/rs_3.jpg'),
#         ('Worker','/images/rs_4.jpg')

class Users(models.Model):
    """  Users Class  """
    user_name = models.CharField(max_length=32, unique=True)
    role = models.ForeignKey(Roles, on_delete=models.CASCADE)
    bio_capsule = models.BinaryField()
    classifier = models.BinaryField()
    created_on = models.DateTimeField(auto_now=True)
    last_authenticated = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user_name
