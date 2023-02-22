import pandas as pd
import numpy as np
import pickle
import os
import re
import scipy
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from collections import defaultdict #data colector

#Surprise: https://surprise.readthedocs.io/en/stable/
import surprise
from surprise.reader import Reader
from surprise import Dataset
from surprise.model_selection import GridSearchCV

##CrossValidation
from surprise.model_selection import cross_validate

##Matrix Factorization Algorithms
from surprise import SVD
from surprise import NMF
from surprise import SVDpp
from surprise import KNNBaseline
from surprise import BaselineOnly

dfs = None
dfx = None
dfp = None

# ------------ 1. VERİ HAZIRLIĞI ----------------------- #
def _convert_tuik(f):
    df = pd.read_excel(f'data/{f}', skiprows=[0,2,3])
    urun_tip = df.iloc[0,0]
    yil = df.iloc[0,2]
    
    dfx = df.drop('Satırlar', axis=1) \
        .drop('Unnamed: 2', axis=1) \
        .rename(columns={df.columns[1]:'Urun'}) \
        .set_index('Urun').T.reset_index().rename(columns={'index':'İlçe'})

    dfx = pd.melt(dfx, id_vars=['İlçe'], value_vars=dfx.columns)
    urun_id = dfx.Urun.apply(lambda x: re.findall('((\d\d\.)+)', x)[0][0])
    urun_olcum = dfx.Urun.apply(lambda x: re.sub('(\d\d\.)+\s','', x).split(' - ')[0].split(' ve ')[0])
    urun_adi = dfx.Urun.apply(lambda x: re.sub('(\d\d\.)+\s','', x).split(' - ')[0].split(' ve ')[1][1:-1])
    olcum_birim = dfx.Urun.apply(lambda x: re.sub('(\d\d\.)+\s','', x).split(' - ')[1])
    ilce_id = dfx.İlçe.apply(lambda x: x.split('-')[1])
    ilce_adi = dfx.İlçe.apply(lambda x: x.split('-')[0].split('(')[1][:-1])
    il_adi = dfx.İlçe.apply(lambda x: x.split('-')[0].split('(')[0])

    dfx = pd.DataFrame.from_dict({
        'Urun_Tipi':urun_tip,
        'Yil':yil,
        'Il_Adi':il_adi,
        'Ilce_Adi':ilce_adi,
        'Ilce_Id':ilce_id,
        'Urun_Id':urun_id, 
        'Urun_Adi':urun_adi,
        'Urun_Olcum':urun_olcum,
        'Olcum_Birim':olcum_birim,
        'Miktar': dfx.value
    })

    return dfx

# 1.1 tuik'den indirilen dataların tablo formatına dönüştürülmesi (dfs)
def create_dataset(path='data', reload=False):
    if reload:
        dfs = [_convert_tuik(f) for f in os.listdir(path) if '.xls' in f]
        dfs = pd.concat(dfs)
        dfs.to_pickle(path+'/tuikdata.pkl')
    else:
        dfs = pd.read_pickle(path+'/tuikdata.pkl')
    return dfs


# ------------ 2. VERİ ÖNİŞLEME ----------------------- #

# 2.1 Çoklayan Verilerin Temizlenmesi (dfs)
def coklama_temizle(dfs):
    """
        tuik'den gelen datada yer alan birbirinin aynısı olan 
        satırların silinmesi
        
        dfs: 1.1 adımında üretilen veri seti
    """
    coklama_adet = dfs.shape[0] - dfs.drop_duplicates().shape[0]
    dfs = dfs.drop_duplicates()
    print(f'{coklama_adet} adet çoklayan kayıt silindi') 
    return dfs


# 2.2. Eksik Verilerin Tamamlanması

# 2.2.1 Süs Biitkileri Verim  (sus_verim)
def eksiveri_sus_verim(dfs):
    """
        tuik datasında yer almayan süs bitkilerine ait verim datasının üretilmesi
        
        dfs: 2.1 adımında üretilen veri seti
    """
    sus_verim = pd.pivot_table(
        dfs[(dfs.Urun_Tipi=='Süs Bitkileri') ] \
            .assign(Urun_Olcum='Verim') \
            .assign(Olcum_Birim2=lambda x: x.Olcum_Birim) \
            .assign(Olcum_Birim='Adet/Metrekare'),
        index=['Urun_Tipi','Yil','Il_Adi','Ilce_Adi','Ilce_Id','Urun_Id','Urun_Adi','Urun_Olcum','Olcum_Birim'],
        columns=['Olcum_Birim2'],
        values='Miktar',
        aggfunc='max'
    ).reset_index()
    sus_verim = sus_verim.assign(Miktar=lambda x: x['Adet Sayısı']/x['Metrekare']).drop(['Adet Sayısı','Metrekare'], axis=1)
    return sus_verim  

# 2.2.2 Sebzeler Verim (sebze_verim)
def eksikveri_sebze_verim(dfs):
    """
        tuik datasında yer almayan sebze türlerine ait verim datasının üretilmesi
        
        dfs: 2.1 adımında üretilen veri seti
    """
    sebze_verim = pd.pivot_table(
        dfs[(dfs.Urun_Tipi=='Sebzeler') ] \
            .assign(Urun_Olcum='Verim') \
            .assign(Olcum_Birim2=lambda x: x.Olcum_Birim) \
            .assign(Olcum_Birim='Ton/Dekar'),
        index=['Urun_Tipi','Yil','Il_Adi','Ilce_Adi','Ilce_Id','Urun_Id','Urun_Adi','Urun_Olcum','Olcum_Birim'],
        columns=['Olcum_Birim2'],
        values='Miktar',
        aggfunc='max'
    ).reset_index()
    sebze_verim = sebze_verim.assign(Miktar=lambda x: x['Ton']/x['Dekar']).drop(['Ton','Dekar'], axis=1)
    return sebze_verim    


# 2.3 Eksikleri Birleştir (dfe)
def eksikleri_birlestir(datasetler):
    """
        tablo formatına dönüştürülen tuik datası ile eksik dataların birleştirilmesi
        adımlar: 2.1, 2.2.1, 2.2.2
        datasetler: [dfs, sebze_verim, sus_verim]
    """
    print('Birleştirilen veri seti adeti:', len(datasetler))
    print('Kayıt adetleri:', ','.join([str(len(d)) for d in datasetler]))
    dfe = pd.concat(datasetler)
    
    print('Eksik değere sahip gözlem adeti', dfe.shape[0] - dfe.dropna().shape[0])
    dfe = dfe.dropna()
    print('Sonuç veri seti gözlem adeti', dfe.shape)
        
    return dfe


# **********************************************************
# 2.4 Ana Tablo Formatının Oluşturulması (dfx) *************
def urun_dataset(dfe, refresh=False, path='data'):
    """
        2.3 adımında üretilen datadaki ürün ve üretim verileri bazındaki tablonun
        her satırda bir ürün bulunan ve üretim bilgilerinin (ekim alan, miktar, verim, ağaç adet, vb)
        sütunlara dönüştürüldüğü format
        
        dfe: 2.3 adımında oluşturulan tablo ve eksik verilerin eklendiği tablo
        refresh: True ise hesaplama işlemi tekrar yapılır, .pkl dosyası olarak kaydedilir
                 False ise hesaplama tekrar yapılmaz, veriler .pkl dosyasından okunur
        path: pkl dosyasının bulunduğu dizin adı
        
        Return: 
            dfx
    """
    def ff(x):
        try:
            ret = None if len(x)==0 else max(x)
            if len(x)>1:
                print('Çoklu:', x)
        except Exception as ex:
            print(x)
            print('Error:',ex)
        return ret
    
    if refresh:
        # hesaplama süresi: ~1.5-2 dk
        dfx = dfe \
            .groupby(['Urun_Tipi','Il_Adi','Ilce_Adi','Urun_Adi']) \
            .apply(lambda x: pd.Series({
                'Ağaç': ff(x[x.Urun_Olcum=='Meyve Veren Yaşta Ağaç Sayısı'].Miktar),
                #'MeyveVermeyenAğaç': ff(x[x.Urun_Olcum=='Meyve Vermeyen Yaşta Ağaç Sayısı'].Miktar),
                'Meyvelik_Alan': ff(x[x.Urun_Olcum=='Toplu Meyveliklerin Alanı'].Miktar),
                'Meyvelik_Alan_Olcu': ff(x[x.Urun_Olcum=='Toplu Meyveliklerin Alanı'].Olcum_Birim),
                'Ekilen_Alan': ff(x[x.Urun_Olcum=='Ekilen Alan'].Miktar),
                'Ekilen_Alan_Olcu': ff(x[x.Urun_Olcum=='Ekilen Alan'].Olcum_Birim),
                'Hasat_Alan': ff(x[x.Urun_Olcum=='Hasat Edilen Alan'].Miktar),
                'Hasat_Alan_Olcu': ff(x[x.Urun_Olcum=='Hasat Edilen Alan'].Olcum_Birim),
                'Uretim_Miktar': ff(x[x.Urun_Olcum=='Üretim Miktarı'].Miktar),
                'Uretim_Olcu': ff(x[x.Urun_Olcum=='Üretim Miktarı'].Olcum_Birim),
                'Verim': ff(x[x.Urun_Olcum=='Verim'].Miktar),
                'Verim_Olcu': ff(x[x.Urun_Olcum=='Verim'].Olcum_Birim),
            })).reset_index()
        
        # her ürünün için yetiştiği ilçedeki toplamın kümülatif toplam içindeki yerini ekler
        dfx = dfx.sort_values(['Urun_Adi','Uretim_Miktar'], ascending=[True, False]) \
                    .assign(Urun_Miktar_CumSum = lambda x: x.groupby(['Urun_Adi'])['Uretim_Miktar'].transform(lambda x: x.cumsum()/x.sum()))
        
        dfx[dfx.Verim.isna()].iloc[:,:]
        dfx.to_pickle(path+'/tubitak_dfx.pkl')
    else:
        dfx = pd.read_pickle(path+'/tubitak_dfx.pkl')
    
    print('Ana tablo oluşturuldu:', dfx.shape)
    return dfx
# **********************************************************

    
# 2.5 Ana Tablo Düzeltilmeleri (dfx)

# 2.5.1 Sıfır üretim ve verime sahip gözlemler
def sifirkayit_duzeltme(dfx):
    # 2.5.1 Sebzeler Ekili Alan 0
    print('Sebzeler - Ekili Alan=0:', dfx[((dfx.Urun_Tipi=='Sebzeler') & (dfx.Ekilen_Alan==0))].shape[0])
    dfx = dfx[~((dfx.Urun_Tipi=='Sebzeler') & (dfx.Ekilen_Alan==0))]
    
    # 2.5.2 Üretim Miktarı > 0, verim=0
    print('Üretim miktarı sıfırdan büyük olup verim oranı 0 olan kayıtlar:', dfx[((dfx.Uretim_Miktar>0) & (dfx.Verim==0))].shape[0])
    dfx = dfx[~((dfx.Uretim_Miktar>0) & (dfx.Verim==0))]
    
    
    # 2.5.3 Üretim Miktarı > 0, kontrol amaçlı adım adım gidilmiştir.
    print('Üretim miktarı sıfırdan büyük olup verim oranı 0 olan kayıtlar:', dfx[((dfx.Uretim_Miktar==0) & (dfx.Verim==0))].shape[0])
    dfx = dfx[~((dfx.Uretim_Miktar==0) & (dfx.Verim==0))]
    
    return dfx

# 2.5.2 Nadir ürünlerin çıkartılması (dfx)
def nadir_urunler(dfx, n=5):
    nadir_urunler = dfx.groupby('Urun_Adi').apply(lambda x: pd.Series({'lokasyon adet':len(x)})).sort_values('lokasyon adet')
    nadir_urunler = nadir_urunler[nadir_urunler['lokasyon adet']<n]
    
    nadir_urunler = nadir_urunler.index.to_list()
    print('Silinen gözlem adet', dfx[dfx.Urun_Adi.isin(nadir_urunler)].shape[0])
    dfx = dfx[~dfx.Urun_Adi.isin(nadir_urunler)]
    return dfx


# 2.6 Veri Doğrulaması (dfa)
def urun_verim_aralik(dfx, urun, ss=3, draw_chart=False):
    df_urun = dfx[dfx.Urun_Adi==urun].Verim
    q1, q3 = df_urun.quantile(.25), df_urun.quantile(.75)
    iqr = round(q3-q1,2)
    drange = round(max(0,q1-1.5*iqr),2), q3+1.5*iqr
    #print('iqr:',iqr, 'range:', drange)
    
    m, s = round(df_urun.mean(),2), round(df_urun.std(),2)
    mrange = (round(max(0,m - ss*s), 2),round(m + ss*s, 2))
    #print('mean:', m, 'range:', mrange)

    if draw_chart:
        ax = df_urun.hist(bins=10)
        ax.axvline(x=m, color='red', linestyle='-.')
        ax.axvline(x=mrange[0], color='red', linestyle='--')
        ax.axvline(x=mrange[1], color='red', linestyle='--')
        [ax.spines[s].set_visible(False) for s in ('left','top','right')]
        ax.set_title(urun)
    """    
    df_urun = df_urun.apply(lambda x:
                            mrange[0] if x<mrange[0] else
                            mrange[1] if x>mrange[1] else
                            x)
    """                       
    return mrange

def tum_aykiri_degerleri_duzelt(dfx, ss=3):
    """
        her bir ürün için urun_verim_aralik fonksiyonu ile 3 standart sapma aralığı hesaplanır
        ve bu aralık dışındaki değerler bu aralığa getirilirek Verim sütunu güncellenir.
        
        Returns:
            dfa
    """
    dfa = dfx.copy()
    urunler = dfx.Urun_Adi.drop_duplicates()
    urun_range = {urun: urun_verim_aralik(dfx=dfx, urun=urun, ss=ss) for urun in urunler}

    def fr(urun, verim):
        urange = urun_range[urun]
        return urange[0] if verim<urange[0] else \
               urange[1] if verim>urange[1] else \
               verim

    dfa['Verim'] = dfa.apply(lambda x: fr(x.Urun_Adi, x.Verim) ,axis=1)    
    
    return dfa


# 2.7 Özet Tablo Oluşturma (dfp)
def ozet_tablo_olustur(dfa, path='data'):
    """
        temizlenmiş ve düzeltilmiş veriler öneri algoritmasına uygun hale getirilir
        Her bir ürüne ait verim değeri ilgili ürünün max değeri ile normalize edilir.
        
        Returns:
            dfp
    """
    dfp = pd.pivot_table(
            dfa, 
            index=['Il_Adi','Ilce_Adi'], 
            columns=['Urun_Adi'], 
            values='Verim', 
            aggfunc='max')
    dfp.to_pickle(path+'/tubitak_dfp.pkl')
    return dfp


# 2.8 Özet tabloyu normalize et (dfn)
def ozet_tablo_normalize(dfp):
    """
        her bir ürünün en yüksek verim değeri ile tüm ilçelerin verim datasının normalize 
        edilmesi
        
        Returns:
            dfn
    """
    dfn = dfp / dfp.max(axis=0)
    return dfn


# 2.9 Normalize tabloyu öneri modeline uygun formata çevir
def ozet_tablo_oneri_format(dfn):
    df_verim = pd.melt(dfn.reset_index(), id_vars=['Il_Adi','Ilce_Adi'], value_vars=dfn.columns).dropna()
    df_verim = df_verim.apply(lambda x: pd.Series({
        'uid': x.Il_Adi+'_'+x.Ilce_Adi, 
        'iid':x.Urun_Adi, 
        'rating': x.value}
    ),axis=1)
    return df_verim        
    

# ------------ 3. MODELLEME  ------------------------------- #

def similarity(dfa, loc1, loc2):
    if loc1==loc2:
        return 0
    else:
        df_sim = dfa[dfa.index.isin([loc1, loc2], level=1)].T.dropna()
        if df_sim.shape[0]<5:
            return 0
        
        #cd = 1-cosine(df_sim.iloc[:,0], df_sim.iloc[:,1])
        if df_sim.shape[1]==2:
            x = df_sim.iloc[:,0]
            y = df_sim.iloc[:,1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_value = 0 if p_value<0.05 else r_value
        else:
            r_value = 0
        
        return r_value #f'{df_sim.shape[0]}-{cd:.2f}'

# model class
class cls_recommend:
    
    def __init__(self):
        """            
            https://www.jiristodulka.com/post/recsys_cf/
            
            http://surpriselib.com/
            Movie 100k		RMSE	MAE	Time
            SVD			    0.934	0.737	0:00:11
            SVD++		    0.92	0.722	0:09:03
            NMF			    0.963	0.758	0:00:15
            Slope One		0.946	0.743	0:00:08
            k-NN		    0.98	0.774	0:00:10
            Centered k-NN	0.951	0.749	0:00:10
            k-NN Baseline	0.931	0.733	0:00:12
            Co-Clustering	0.963	0.753	0:00:03
            Baseline		0.944	0.748	0:00:01
            Random		1.514	1.215	0:00:01            
        """
        
    def set_data(self, df, rating_scale=(0.1, 1)):
        """
            ham datayı öneri sisteminin okuyacağı formata çevirir
        """
        reader = Reader(rating_scale=rating_scale)
        self.data = Dataset.load_from_df(df, reader=reader)
        self.trainset = self.data.build_full_trainset()
        self.testset = self.trainset.build_anti_testset()
        return self.data
            
    def rmse_vs_factors(self, algorithm, data, num_factors):
        """
        Arg:  
            algorithm:
                Matrix factoization algorithm, e.g SVD/NMF/PMF, 
            data:
                surprise.dataset.DatasetAutoFolds
                
        Returns: 
            rmse_algorithm i.e. a list of mean RMSE of CV = 5 in cross_validate() 
            for each  factor k in range(1, 101, 1) 100 values 
        """
        rmse_results = []

        for k in range(1, num_factors, 1):
            algo = algorithm(n_factors = k)

            #["test_rmse"] is a numpy array with min accuracy value for each testset
            loss_fce = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)["test_rmse"].mean() 
            rmse_results.append(loss_fce)
            print('.', end='')
            if k%10==0: print('|', end='')

        return rmse_results

    def plot_rmse(self, rmse_results, algorithm):
        """Returns: sub plots (2x1) of rmse against number of factors. 
         Vertical line in the second subplot identifies the arg for minimum RMSE

         Arg: i.) rmse = list of mean RMSE returned by rmse_vs_factors(), ii.) algorithm = STRING! of algo 
        """

        plt.figure(num=None, figsize=(11, 5), dpi=80, facecolor='w', edgecolor='k')

        plt.subplot(2,1,1)
        plt.plot(rmse_results)
        plt.xlim(0, len(rmse_results) )
        plt.title("{0} Performance: RMSE Against Number of Factors".format(algorithm), size = 12 )
        plt.ylabel("Mean RMSE (cv=5)")
        plt.xlabel("{0}(n_factor = k)".format(algorithm))
        plt.axvline(np.argmin(rmse_results), color = "r")
        plt.show()
        
    def run_test(self, algorithm, df, num_factors=5):
        data = self.set_data(df)
        self.rmse_results = self.rmse_vs_factors(algorithm, data, num_factors)
        self.plot_rmse(self.rmse_results, algorithm)
        
    def find_best_param(self, algorithm, df, rating_scale, n_factors, verbose=False):
        data = self.set_data(df, rating_scale)
        param_grid = {'n_factors': n_factors}
        gs = GridSearchCV(algorithm, param_grid, measures=['rmse'], cv=5, n_jobs=5, joblib_verbose=verbose)
        gs.fit(data)

        # best RMSE score
        print(gs.best_score['rmse'])

        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])
        
        self.best_param = gs.best_params['rmse']
        
    def train(self, df, algorithm_name='NMF', rating_scale=(0,1), n_factors=10):
        # http://surpriselib.com/
        
        if algorithm_name=='NMF':
            algorithm = NMF
            self.model = algorithm(n_factors = n_factors)
        elif algorithm_name=='SVD':
            algorithm = SVD
            self.model = algorithm(n_factors = n_factors)
        elif algorithm_name=='SVDpp':
            algorithm = SVDpp
            self.model = algorithm(n_factors = n_factors)
            
        elif algorithm_name=='KNNBaseline':
            algorithm = KNNBaseline
            self.model = algorithm()
        else:
            algorithm = BaselineOnly
            self.model = algorithm()
            
        
        data = self.set_data(df)
        self.model.fit(self.trainset)

        # Predict ratings for all pairs (i,j) that are NOT in the training set.
        self.testset = self.trainset.build_anti_testset()

        predictions = self.model.test(self.testset)
        df_pred = pd.DataFrame(predictions).iloc[:, [0,1,3]]
        df_pred = df_pred \
            .assign(rank=df_pred.groupby('uid')['est'].rank(method='first', ascending=False))\
            .sort_values(['uid','rank'], ascending=[True,False]) \
            .sort_values('est', ascending=False)
        self.predictions = df_pred
        
        return self.model
    
        
    def algorithm_backtest(self, algorithm_name, df, rating_scale, n_factors, 
                           test_size=1000, test_treshold=0.1,
                           display_chart=False, verbose=False):
        self.df = df
        df_a = df.sample(test_size)
        df_b = df.drop(df_a.index)
        self.train(df_b, algorithm_name=algorithm_name, rating_scale=rating_scale, n_factors=n_factors)
        
        df_pred = self.predictions
        df_compare = df_a.merge(df_pred, 
                                how='left', 
                                left_on=['uid','iid'], 
                                right_on=['uid','iid']) \
            .assign(Error_Net = lambda x: round((x.est-x.rating),1)) \
            .assign(Error_Ratio = lambda x: round((x.est-x.rating)/x.rating,1)) \
            .query("""(est>=0.0)""")

        r2 = r2_score(df_compare.rating, 
                      df_compare.est)
        cvr_net = df_compare[df_compare.Error_Net.abs()<=test_treshold].shape[0] / df_compare.shape[0]
        cvr_rat = df_compare[df_compare.Error_Ratio.abs()<=test_treshold].shape[0] / df_compare.shape[0]
        
        ret = f'{algorithm_name}: R^2:{r2:.3f}, Net Kapsama:{cvr_net*100:.1f}%, Oran Kapsama:{cvr_rat*100:.1f}%'
        
        if display_chart:
            f = 'Net'
            cvr = cvr_net
            dx = df_compare.query(f"""(Error_{f}>-2) and (Error_{f}<2)""")[f'Error_{f}'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(12,5))
            ax.bar(x=     dx.index, height=dx.values, width=0.08, edgecolor='w' )
            ax.bar(x=     dx[(dx.index>=-test_treshold) & (dx.index<=test_treshold)].index, 
                   height=dx[(dx.index>=-test_treshold) & (dx.index<=test_treshold)].values, 
                   width=0.08, color='g', edgecolor='w' )
            ax.set_xticks(dx.index)
            [ax.spines[s].set_visible(False) for s in ('left','top','right')]
            ax.grid(True, axis='y')
            ax.set_title(f'{algorithm_name} - R^2:{r2:.3f}, <>:%{cvr*100:.1f}')
            plt.show()        
        
        self.df_backtest = df_compare
        return df_compare
    
    def run_all(self, data_path, algorithm_name='KNN', n_factors=50, test_size=10000, n_nadir_urun=20, backtest_treshold=0.1):
        dfs = self.create_dataset(path=data_path, reload=False)
        dfs = self.coklama_temizle(dfs)
        sus_verim = self.eksiveri_sus_verim(dfs)
        sebze_verim = self.eksikveri_sebze_verim(dfs)
        dfe = self.eksikleri_birlestir([dfs, sus_verim, sebze_verim])
        dfx = self.urun_dataset(dfe, refresh=True, path=data_path)
        dfx = self.sifirkayit_duzeltme(dfx)
        dfx = self.nadir_urunler(dfx, n=n_nadir_urun)
        #dfx = dfx[~dfx.Urun_Adi.isin(['Çay Yaprakları'])]
        #dfx = dfx[~dfx.Il_Adi.isin(['Iğdır'])]
        dfa = self.tum_aykiri_degerleri_duzelt(dfx,3)
        dfp = self.ozet_tablo_olustur(dfa, path=data_path)
        dfn = self.ozet_tablo_normalize(dfp)
        df_verim = self.ozet_tablo_oneri_format(dfn)

        df_verim_filter = df_verim.merge(
            dfx.assign(**{'uid': lambda x: x.apply(lambda y: y.Il_Adi+'_'+y.Ilce_Adi, axis=1),
                        'iid': lambda x: x.Urun_Adi
                        }).loc[:,['uid','iid','Urun_Miktar_CumSum']],
            on=['uid','iid']
        ).query('Urun_Miktar_CumSum<0.98').loc[:,['uid','iid','rating']]             
    
        cls_rcm = self.cls_recommend()
        df_bt = cls_rcm.algorithm_backtest(
            algorithm_name=algorithm_name,
            df=df_verim_filter, 
            rating_scale=(0,1), 
            n_factors=n_factors,
            test_size=test_size, 
            test_treshold=backtest_treshold,
            display_chart=True
        )

        # ********* df_final *******************************************************

        columns = ['Il_Adi','Ilce_Adi','Urun_Tipi','Urun_Adi',
                'Uretim_Miktar','Uretim_Olcu', 'Verim', 'Verim_Olcu',
                'prediction','rating','rating_actual']
        df_final = \
            pd.concat([
                    # train datası ile predict datası birleştirilir 
                    cls_rcm.df.assign(prediction=0),
                    cls_rcm.predictions.drop('rank', axis=1).rename(columns={'est':'rating'}).assign(prediction=1)
                ]) \
            .assign(
                # train-predict tablosunda uid olarak tutulan il-ilçe adlarını ayrıştır
                    **{'Il_Adi': lambda x: x.uid.apply(lambda x: x.split('_')[0]),
                    'Ilce_Adi': lambda x: x.uid.apply(lambda x: x.split('_')[1])
                    }) \
            .merge(
                    # sonuç tablosu ile ürün adlarını birleştir,
                    # Ürün Tipi'ni buradan alıyoruz
                    dfs[['Urun_Tipi','Urun_Adi']].drop_duplicates(),
                    how='left',
                    left_on='iid', right_on='Urun_Adi',
                    suffixes=['_a',None]
                ) \
            .merge(
                # oluşan sonuç tablosu ile verimlilik tablosunu birleştir, 
                # train-predict datasındaki ratinglerle gerçek datadaki ratingleri yanyana getir 
                df_verim,
                how='left',
                left_on=['iid','uid'], right_on=['iid','uid'],
                suffixes=[None,'_actual'],
                ) \
            .merge(
                    # train-predict sonuç tablosu ile ürün bazlı pivot tabloyu birleştir
                    # üretim miktarı, üretim birimi, gerçek verim miktarını bu tablodan alıyoruz
                    dfx.iloc[:,[0,1,2,3,11,12,13,14]],
                    how='left',
                    left_on=['iid','Il_Adi','Ilce_Adi'], right_on=['Urun_Adi','Il_Adi','Ilce_Adi'],
                    suffixes=[None, '_x']
                ).drop(['Urun_Tipi_x','Urun_Adi_x'], axis=1) \
            [columns]

        df_final.to_pickle('../app/static/data/df_final.pkl')

        self.dfx = dfx
        self.df_verim = df_verim
        self.df_bt = df_bt
        self.df_final = df_final

        return df_final
    
# ********** YARDIMCI FONKSİYONLAR ***************************
def my_tabulate(df):
    """
        bir veri setini markdown formatına dönüştürür
    """
    
    head = '|'+ '|'.join(df.columns) + '|'
    head2 = '|'+ '|'.join(['-' for i in df.columns]) + '|'
    rows = df.apply(lambda x: f'|{"|".join([str(c) for c in x])}|' ,axis=1).values
    rows = '\n'.join(rows)
    
    print(head)
    print(head2)
    print(rows)
    
    