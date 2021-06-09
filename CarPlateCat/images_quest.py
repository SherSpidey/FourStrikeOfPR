import re  # 导入正则表达式模块
import requests


def get_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 '
                      'Safari/537.36 SE 2.X MetaSr 1.0 '
    }
    reponse = requests.get(url, headers=headers)
    reponse.encoding = reponse.apparent_encoding
    html = reponse.text
    return html


def get_img(html):
    # [^"]+\.jpg 匹配除"以外的所有字符多次,后面跟上转义的.和png
    P = [r'<img src="([^"]+\.jpeg)"',
         r'<img src="([^"]+\.png)"',
         r'<img src="([^"]+\.jpg)"',
         r'<img src=\'([^"]+\.jpeg)\'',
         r'<img src=\'([^"]+\.png)\'',
         r'<img src=\'([^"]+\.jpg)\'',
         r'src="([^"]+\.jpeg)"',
         r'src="([^"]+\.png)"',
         r'src="([^"]+\.jpg)"',
         ]
    # 返回正则表达式在字符串中所有匹配结果的列表
    i = 48
    for p in P:
        img_list = re.findall(p, html)
        print(img_list)

        # 循环遍历列表的每一个值
        for each in img_list:
            # 以/为分隔符，-1返回最后一个值
            part = each.split('/')[0]
            if part != 'https:' and part != 'http:':
                each = 'https://' + each
            filename = "" + str(i)

            # 访问each，并将页面的二进制数据赋值给photo
            try:
                photo = requests.get(each)
            except:
                continue
            else:
                w = photo.content
                # 打开指定文件，并允许写入二进制数据
                f = open('D:/images/get/' + filename + '.png', 'wb')
                # 写入获取的数据
                f.write(w)
                # 关闭文件
                f.close()
                i += 1


# 爬取网络上的汽车车牌图片
if __name__ == '__main__':
    # 定义url
    url = ""
    # 将url作为open_url()的参数，然后将open_url()的返回值作为参数赋给get_img()
    get_img(get_html(url))
