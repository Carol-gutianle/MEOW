import requests
import json

class Robot:
    def __init__(self):
        self.url = 'https://open.feishu.cn/open-apis/bot/v2/hook/{YOUR HOOK}'

    def post_message(self, text):
        msg = {
            'text': text
        }
        body = json.dumps({"msg_type": "text", "content": msg})
        headers = {"Content-Type": "application/json"}
        res = requests.post(url=self.url, data=body, headers=headers)
        print(res)