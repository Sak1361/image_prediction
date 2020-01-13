import json
import requests
import os
import urllib
import imghdr
import sys
from tqdm import tqdm
from bs4 import BeautifulSoup
import pykakasi  # 全角を半角ローマ字に変換


class Google:
    def __init__(self):
        self.GOOGLE_SEARCH_URL = 'https://www.google.co.jp/search'
        self.session = requests.session()
        self.session.headers.update(
            {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0'})

    def Search(self, keyword, type='text', maximum=100):
        print('Google', type.capitalize(), 'Search :', keyword)  # Google検索
        result, total = [], 0
        query = self.query_gen(keyword, type)
        while True:  # 検索
            html = self.session.get(next(query)).text
            links = self.get_links(html, type)
            if not len(links):  # 検索結果の追加
                print('-> No more links')
                break
            elif len(links) > maximum - total:
                result += links[:maximum - total]
                break
            else:
                result += links
                total += len(links)
        print('-> Finally got', str(len(result)), 'links')
        return result

    def query_gen(self, keyword, type):  # 検索クエリジェネレータ
        page = 0
        while True:
            if type == 'text':
                params = urllib.parse.urlencode({
                    'q': keyword,
                    'num': '100',
                    'filter': '0',
                    'start': str(page * 100)})
            elif type == 'image':
                params = urllib.parse.urlencode({
                    'q': keyword,
                    'tbm': 'isch',
                    'filter': '0',
                    'ijn': str(page)})

            yield self.GOOGLE_SEARCH_URL + '?' + params
            page += 1

    def get_links(self, html, type):  # リンク取得
        soup = BeautifulSoup(html, 'lxml')
        if type == 'text':
            elements = soup.select('.rc > .r > a')
            links = [e['href'] for e in elements]
        elif type == 'image':
            elements = soup.select('.rg_meta.notranslate')
            jsons = [json.loads(e.get_text()) for e in elements]
            links = [js['ou'] for js in jsons]
        return links

    def name_index(self, basename, digits=3, first=0):  # インデックス付きで名前生成
        i = first
        style = '%0' + str(digits) + 'd'
        while True:
            yield basename + '_' + str(style % i)
            i += 1

    def imageExt(self, b_content):  # 画像形式の判定
        ext = imghdr.what(None, b_content)
        if ext is None and b_content[:2] == b'\xff\xd8':
            ext = 'jpeg'
        return ext

    def save_images(self, url_list, basename, dir='./', digits=3):  # 画像をインデックス付きでまとめて保存
        dir = os.path.join(dir, basename)
        os.makedirs(dir, exist_ok=True)
        name = google.name_index(basename, digits, first=1)  # ファイル名ジェネレータ
        for url in tqdm(url_list):
            url = urllib.parse.unquote(url)  # urlの文字長エラー回避
            # アクセス
            try:
                content = requests.get(url).content
            except Exception as ex:
                print(url, '\n->', ex)
                continue
            # 画像形式を判定
            ext = google.imageExt(content)
            if ext is None:
                ext = os.path.splitext(url)[1][1:]
            # 保存
            filename = os.path.join(dir, next(name) + '.' + ext)
            with open(filename, 'wb') as f:
                f.write(content)


def kakasi(words):
    kakasi = pykakasi.kakasi()
    kakasi.setMode('H', 'a')
    kakasi.setMode('K', 'a')
    kakasi.setMode('J', 'a')
    conv = kakasi.getConverter()

    return conv.do(words)


if __name__ == "__main__":
    google = Google()
    #word = input("検索する単語：")
    word = sys.argv[1]
    print(word)

    maximum = int(input("何件取ってくる？："))
    typeselect = int(input("Text?[0] or Image?[1]："))
    if typeselect:
        typeselect = "image"
    else:
        typeselect = "text"
    # テキスト検索
    #result = google.Search(word, type='text', maximum=200)
    # 画像検索
    url_list = google.Search(word, type=typeselect, maximum=maximum)
    result = google.save_images(
        url_list, basename=kakasi(word) + "-" + typeselect)
    print("にゃーん")
