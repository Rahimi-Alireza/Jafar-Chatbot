def get_data():
    return ["intents": [
        ["tag": "open",
         "patterns": ["باز کن", "برام باز کن", "میتونی باز کنی", "باز میکنی"],
         "Action": "open",
         "context_set": ""
        ],

        ["tag": "close",
         "patterns": ["ببند", "ببندش", "میتونی ببندی"],
         "Action": "close",
        ]
   ]
]
