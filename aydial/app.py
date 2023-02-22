from flask import Flask, json, render_template, render_template_string, redirect, url_for, jsonify, request, session
from flask_session import Session
import random
import pandas as pd
import time
import zipfile
import datetime as dt
import numpy as np
import pickle as pkl

app = Flask(__name__,
            static_folder='static',
            template_folder='templates'
            )

app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = "filesystem"
Session(app)

data_path = 'static/data/'
df_final = None
df_ozet = None
df_urunbilgi = None
tahmin_urun_list = None


@app.route('/aciklama')
def aciklama():
    return render_template('aciklama.html')


@app.route('/loaddata')
def load_data():
    global df_final, df_ozet, df_urunbilgi, data_path, tahmin_urun_list
    # path = url_for('static', filename='data/df_final.pkl')
    df_final = pd.read_pickle(data_path + 'df_final.pkl')
    df_ozet = pd.read_pickle(data_path + 'df_ozet.pkl')

    # ürün listesi için böyle bir filtre koymuşuz, nedenini hatırlayamadım [(df_ozet.Urun_Adi.str.contains('a'))]
    tahmin_urun_list = df_ozet.groupby('Urun_Adi').miktar.sum().sort_values(ascending=False).iloc[:80].index.to_list()

    df_urunbilgi = pd.DataFrame([],
                                columns=['Urun_Adi', 'Urun_Aciklama'])  # pd.read_pickle(data_path+'df_urunbilgi.pkl')
    print('df_final', df_final.shape)
    print('df_ozet', df_ozet.shape)
    print('df_urunbilgi', df_urunbilgi.shape)
    print('tahmin_list', len(tahmin_urun_list))
    return render_template_string(f'veriler yüklendi: df_ozet:{df_ozet.shape[0]}, df_final: {df_final.shape[0]}')


def counter(method='get'):
    with open('static/data/counter.txt', 'r') as f:
        c = int(f.read())
    if method == 'set':
        c += 1
        with open('static/data/counter.txt', 'w') as f:
            f.write(str(c))
    return c


@app.route('/')
@app.route('/home')
def home():
    c = counter('set')
    load_data()
    return render_template('harita.html', counter=c)


@app.route('/get_urunler')
def get_urunler():
    # df_uruntipi = df_ozet['Urun_Tipi'].drop_duplicates().sort_values().values
    # df_uruntipi = pkl.load(open(data_path+'df_uruntipi.pkl','rb'))
    # col_rename_map = dict(zip(df_uruntipi,
    #                          ['Meyveler', 'Sebzeler', 'Süs Bitkileri', 'Tahıllar']))

    #df_urunler = df_ozet[['Urun_Tipi', 'Urun_Adi']].drop_duplicates() \
    #    .sort_values(['Urun_Tipi', 'Urun_Adi']).groupby('Urun_Tipi').agg(list).T \
    #    .rename(columns=col_rename_map).T

    df_urunler = pkl.load(open(data_path + 'df_urunler.pkl', 'rb'))
    urunler = df_urunler.to_dict(orient='dict')['Urun_Adi']
    return jsonify({'urunler': urunler})


@app.route('/get_iller')
def get_iller():
    df_iller = pkl.load(open(data_path+'df_iller.pkl', 'rb'))
    iller = df_iller.apply(lambda x: f"<option value='{x}'>{x}</option>").values.tolist()
    return jsonify({'iller': iller})


@app.route('/urun_bilgi_ekle')
def urun_bilgi_ekle():
    global df_urunbilgi, data_path
    urun = request.args.get('urun')
    bilgi = request.args.get('bilgi')
    df = pd.DataFrame([[urun, bilgi]], columns=df_urunbilgi.columns)
    df_urunbilgi = df_urunbilgi[df_urunbilgi.Urun_Adi != urun].append(df, ignore_index=True)
    df.to_pickle(data_path + 'df_urunbilgi.pkl')
    print('urun_bilgi:', df.shape)
    return jsonify({'ret': 'ok'})


@app.route('/urun_bilgi_oku')
def urun_bilgi_oku():
    #global df_urunbilgi, data_path
    urun = request.args.get('urun')
    df_urunbilgi  = pkl.load(open(data_path+'df_urunbilgi.pkl','rb'))
    bilgi = df_urunbilgi[df_urunbilgi.Urun_Adi == urun].Urun_Aciklama.values.tolist()
    print('urun bilgi', bilgi)
    return jsonify({'ret': bilgi})


@app.route('/getil')
def get_il():
    global df_ozet
    il = request.args.get('il', 'xxx')
    mode = request.args.get('mode', 'a')

    if isinstance(df_ozet, type(None)):
        load_data()
        print('data loaded, from:get_il()')

    dfo = df_ozet.query(f"""(Il_Adi=='{il}') & (Urun_Tipi=='Meyveler Içecek Ve Baharat Bitkileri')""")
    meyveler = dfo.sort_values(['verim_' + mode], ascending=False).iloc[:10, 2:10].fillna(0).values.tolist()
    dfo = df_ozet.query(f"""(Il_Adi=='{il}') & (Urun_Tipi=='Tahıllar Ve Diğer Bitkisel Ürünler')""")
    tahillar = dfo.sort_values(['verim_' + mode], ascending=False).iloc[:10, 2:10].fillna(0).values.tolist()
    dfo = df_ozet.query(f"""(Il_Adi=='{il}') & (Urun_Tipi=='Süs Bitkileri')""")
    susbitkileri = dfo.sort_values(['verim_' + mode], ascending=False).iloc[:10, 2:10].fillna(0).values.tolist()
    dfo = df_ozet.query(f"""(Il_Adi=='{il}') & (Urun_Tipi=='Sebzeler')""")
    sebzeler = dfo.sort_values(['verim_' + mode], ascending=False).iloc[:10, 2:10].fillna(0).values.tolist()

    return jsonify({'ret': il,
                    'meyveler': meyveler,
                    'tahillar': tahillar,
                    'sebzeler': sebzeler,
                    'susbitkileri': susbitkileri
                    })


@app.route('/il_detay')
def il_detay():
    il = request.args.get('il', 'xxx')
    urun = request.args.get('urun', 'xxx')

    if isinstance(df_final, type(None)):
        load_data()
        print('data loaded, from:il_detay()')

    # 'rating_actual':'Gerçek Verim'
    df = df_final[(df_final.Il_Adi == il) & ((df_final.Urun_Adi == urun) | (urun == ''))] \
        .sort_values(['prediction', 'rating'], ascending=[True, False]) \
        .loc[:,
             ['Il_Adi', 'Ilce_Adi', 'Urun_Adi', 'Uretim_Miktar', 'Uretim_Olcu', 'Verim', 'Verim_Olcu', 'prediction',
              'rating', ]] \
        .rename(columns={'prediction': 'Tahmin', 'rating': 'Verim Oranı', }) \
        .assign(Tahmin=df_final.prediction.replace({1: 'tahmin', 0: 'üretilen'}))
    tahmin = pd.IndexSlice[df.loc[(df['Tahmin'] == 'tahmin')].index, 'Verim Oranı']
    gercek = pd.IndexSlice[df.loc[(df['Tahmin'] == 'üretilen')].index, 'Verim Oranı']

    return jsonify({
        'tablo1': df.to_html(index=False, classes='mystyle'),
        'tablo2': df.style.format('{:,.2f}', subset=['Uretim_Miktar', 'Verim', 'Verim Oranı'])
        .applymap(lambda x: "color:rgb(5, 50, 80)", subset=gercek)
        .bar(color="rgba(21, 172, 205, 0.8)", subset=gercek, vmin=0, vmax=1)
        .bar(color="rgba(10, 160,  80, 0.4)", subset=tahmin, vmin=0, vmax=1)
        .hide_index()
        .render().replace('nan', '')
    })


# ******************* OYUN *********************************************************************************************

@app.route('/tahminet', methods=['POST', 'GET'])
def tahmin_et():
    json_data = request.get_json()
    il = json_data['il']
    urunler = json_data['urunler']
    df = df_ozet[(df_ozet.Il_Adi == il) & (df_ozet.Urun_Adi.isin(urunler))].iloc[:, :].set_index('Urun_Adi')
    # print(df)
    return jsonify({'urunler': df.fillna(0).T.to_dict(orient='dict')})


# oyun 2

@app.route('/tahmin_urun_al', methods=['POST', 'GET'])
def tahmin2_urun_al():
    json_data = request.get_json()
    il = json_data['il']
    df1 = df_ozet[(df_ozet.Il_Adi == il) & (df_ozet.Urun_Adi.isin(tahmin_urun_list)) & (df_ozet.miktar > 0)].sample(5)
    df2 = df_ozet[(df_ozet.Il_Adi == il) & (df_ozet.Urun_Adi.isin(tahmin_urun_list)) & (df_ozet.miktar.isna())]
    df2 = df2.sample(min(5, df2.shape[0]))
    df = pd.concat([df1, df2])

    return jsonify({'urunler': df.fillna(0).T.to_dict(orient='dict')})


# ******************* /OYUN ********************************************************************************************

@app.route('/urun_heatmap', methods=['POST', 'GET'])
def urun_heatmap():
    global df_ozet
    if isinstance(df_ozet, type(None)):
        load_data()
        print('data loaded, from:urun_heatmap()')

    json_data = request.get_json()
    urun = json_data['urun']
    df = df_ozet[(df_ozet.Urun_Adi == urun)] \
        .sort_values('ort', ascending=False) \
        .set_index('Il_Adi')
    # print(df)
    return jsonify({
        'urunler': df.fillna(0).T.to_dict(orient='dict'),
        'ozet': df.reset_index().sort_values('ort', ascending=False).fillna(0).values.tolist()
    })


saat_sonuclar = []


@app.route('/saat')
def saat():
    global saat_sonuclar

    tmp = """
        <html lang="en-US">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=yes" >
              <title>Bitkisel Üretim Tahmin Uygulaması</title>

              <link rel="stylesheet" type="text/css" href="../static/css/bootstrap.min.css" >
              <script type="text/javascript" src="../static/js/jquery-1.8.0.min.js"></script>
              <script type="text/javascript" src="../static/js/bootstrap.bundle.min.js"></script>
              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css">
            </head>
            <body>
                <style>
                    table.dataframe, .dataframe th, .dataframe td {{
                      border: none;
                      border-bottom: 1px solid #C8C8C8;
                      border-collapse: collapse;
                      text-align:left;
                      padding: 10px;
                      margin-bottom: 40px;
                      font-size: 0.9em;
                    }}

                    th {{
                        background-color: #557;
                        color: white;
                        height:30px;
                    }}

                    tr:nth-child(odd)		{{ background-color:#eee; }}
                    tr:nth-child(even)	{{ background-color:#fff; }}

                    tr:hover            {{ background-color: #ffff99;}}             
                </style>
                <section id="hero" class="d-flex">
                    <div class="container position-relative">

                      <div class="row justify-content-center mt-3" >
                        <div class="col-xl-7 col-md-12 text-center">
                          <h1>Saat Hesaplama</h1>
                        </div>
                        <div>                      
                            <form method="get">
                                <div class="row g-3">
                                  <div class="col-sm-2 offset-sm-3">
                                    <div class="input-group mb-3">
                                      <input type="text" class="form-control" name="h" value="{h:02d}:{m:02d}">
                                      <div class="input-group-prepend">
                                        <span class="input-group-text" id="basic-addon1">+</span>
                                      </div>
                                    </div>  
                                  </div>
                                  <div class="col-sm-2">
                                    <div class="input-group mb-3">
                                      <input type="text" class="form-control" name="a" value="{a}">
                                      <div class="input-group-prepend">
                                        <span class="input-group-text" id="basic-addon1">dk</span>
                                      </div>
                                    </div>  
                                  </div>
                                  <div class="col-sm-2">
                                    <div class="input-group mb-3">
                                      <div class="input-group-prepend">
                                        <span class="input-group-text" id="basic-addon1">=</span>
                                      </div>
                                      <input type="time" class="form-control" name="r" 
                                          value="00:00" step="0" autofocus>
                                      <input type="text" class="form-control d-none" name="t" value="{r}">
                                      <input type="text" class="form-control d-none" name="z" value="{z}">
                                    </div>  
                                  </div>
                                  <div class="col-sm-2">
                                      <input type="submit" class="btn btn-secondary" value="Gönder" />
                                      <label for="reset">Reset</label>
                                      <input type="checkbox" name="reset" onclick="this.form.submit()" />
                                  </div>
                                </div>                               
                            </form>

                        </div>
                      </div>
                      <div class="row justify-content-center mt-3" >
                          {sonuc}
                      </div>
                      <div class="row justify-content-center mt-3" >
                          {ret}
                      </div>

                    </div>
                </section>
            </body>
        </html>
    """
    h = np.random.randint(1, 23)
    m = np.random.randint(0, 59)
    a = np.random.randint(5, 59)
    t1 = dt.datetime.strptime(f'{h}:{m}', '%H:%M')
    t2 = t1 + dt.timedelta(minutes=a)
    r = t2.strftime('%H:%M')
    z = time.time()

    rh = request.args.get('h')
    if not isinstance(rh, type(None)):
        ra = int(request.args.get('a'))
        rr = request.args.get('r')
        rt = (dt.datetime.strptime(f'{rh}', '%H:%M') + dt.timedelta(minutes=ra)).strftime('%H:%M')
        rc = "<i class='bi bi-emoji-angry text-danger'></i>" if rr != rt \
            else "<i class='bi bi-emoji-smile text-success'></i>"
        rci = 0 if rr != rt else 1
        rz = (round(time.time() - float(request.args.get('z')), 0))
        ret = [rh, ra, rr, rt, rc, rz, rci]

        saat_sonuclar.append(ret)

    if request.args.get("reset"):
        saat_sonuclar = []

    df_ret = pd.DataFrame(saat_sonuclar,
                          columns=['Başlangıç Zaman', 'Artı Dakika', 'Cevap', 'Sonuç', 'Kontrol', 'Zaman',
                                   'Kontrol_int'])
    if df_ret.shape[0] > 0:
        sonuc = f"""
            <span style="font-size:18">
            Başarılı: {df_ret.Kontrol_int.sum()} - {df_ret.Kontrol_int.sum() / df_ret.shape[0] * 100:.0f}%
            Süre: {df_ret.Zaman.mean():.2f}
            </span>
        """
    else:
        sonuc = ""

    df_rets = df_ret.iloc[:, :-1].style.format('{:,.2f}', subset=pd.IndexSlice[:, ['Zaman']]).render(index=False)
    return render_template_string(tmp.format(h=h, m=m, a=a, r=r, z=z,
                                             ret=df_rets, sonuc=sonuc))
