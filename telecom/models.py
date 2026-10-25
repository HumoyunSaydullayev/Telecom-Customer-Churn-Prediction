from django.db import models


class ChurnRecord(models.Model):
    """Mijoz churn yozuvi (ixtiyoriy - bazaga saqlash uchun)"""
    customer_id     = models.CharField(max_length=50, unique=True)
    gender          = models.CharField(max_length=10)
    senior_citizen  = models.BooleanField(default=False)
    tenure          = models.IntegerField()
    contract        = models.CharField(max_length=50)
    monthly_charges = models.FloatField()
    total_charges   = models.FloatField()
    internet_service = models.CharField(max_length=50)
    payment_method  = models.CharField(max_length=100)
    churn           = models.BooleanField(default=False)
    predicted_churn = models.BooleanField(null=True, blank=True)
    churn_probability = models.FloatField(null=True, blank=True)
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Mijoz churn yozuvi"
        verbose_name_plural = "Mijoz churn yozuvlari"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.customer_id} - {'Churn' if self.churn else 'Active'}"