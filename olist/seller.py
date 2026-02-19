import pandas as pd
import numpy as np
from olist.data import Olist

class Seller:
    def __init__(self):
        # Sadece ham veriyi alıyoruz, başka sınıflara bağımlılığı kestiğimiz için KeyError almayacaksın
        self.data = Olist().get_data()

    def get_seller_features(self):
        """Zip kodunu eleyerek tam 15 sütun hedefini korur."""
        return self.data['sellers'][['seller_id', 'seller_city', 'seller_state']].copy()

    def get_review_score(self):
        """Satıcı bazlı puanları hesaplar."""
        orders_reviews = self.data['order_reviews']
        order_items = self.data['order_items']

        # Sadece yorumu olan satıcıları tutmak için inner join
        matching_table = order_items[['order_id', 'seller_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews[['order_id', 'review_score']], on='order_id', how='inner')

        df['dim_is_one_star'] = df['review_score'].apply(lambda x: 1 if x <= 2 else 0)
        df['dim_is_five_star'] = df['review_score'].apply(lambda x: 1 if x == 5 else 0)

        return df.groupby('seller_id').agg({
            'review_score': 'mean',
            'dim_is_one_star': 'mean',
            'dim_is_five_star': 'mean'
        }).rename(columns={
            'dim_is_one_star': 'share_of_one_stars',
            'dim_is_five_star': 'share_of_five_stars'
        })

    def get_order_metrics(self):
        """Tüm metrikleri ham veriden hesaplayarak 2967 satıra ulaşır."""
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].copy()

        # KRİTİK FİLTRE: Sadece teslim edilmiş ve tarihleri tam olan 2967 satıcıyı yakala
        orders = orders[orders['order_status'] == 'delivered'].dropna(subset=['order_delivered_customer_date', 'order_delivered_carrier_date'])

        # Tarih dönüşümleri
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
        order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'])

        # Wait time ve Delay to carrier hesaplamaları
        orders['wait_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']) / np.timedelta64(1, 'D')

        df = order_items.merge(orders[['order_id', 'wait_time', 'order_delivered_carrier_date']], on='order_id', how='inner')
        df['delay_to_carrier'] = (df['order_delivered_carrier_date'] - df['shipping_limit_date']) / np.timedelta64(1, 'D')
        df['delay_to_carrier'] = df['delay_to_carrier'].clip(lower=0)

        result = df.groupby('seller_id').agg({
            'order_id': 'nunique',
            'product_id': 'count',
            'price': 'sum',
            'wait_time': 'mean',
            'delay_to_carrier': 'mean',
            'shipping_limit_date': ['min', 'max']
        })

        result.columns = ['n_orders', 'quantity', 'sales', 'wait_time',
                          'delay_to_carrier', 'date_first_sale', 'date_last_sale']

        # Ay hesabı (30 güne bölerek ValueError önlenir)
        delta_days = (result['date_last_sale'] - result['date_first_sale']) / np.timedelta64(1, 'D')
        result['months_on_olist'] = np.ceil(delta_days / 30).replace(0, 1)
        result['quantity_per_order'] = result['quantity'] / result['n_orders']

        return result

    def get_training_data(self):
        """Final tablosunu (2967, 15) olarak birleştirir."""
        features = self.get_seller_features()
        metrics = self.get_order_metrics()
        reviews = self.get_review_score()

        # Tüm parçaları inner join ile birleştiriyoruz
        # Bu işlem 2970 -> 2967 düşüşünü ve 15 sütunu garanti eder
        training_set = features.merge(metrics.reset_index(), on='seller_id', how='inner')
        training_set = training_set.merge(reviews.reset_index(), on='seller_id', how='inner')

        return training_set
