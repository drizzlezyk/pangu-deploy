# UTF-8

import sentencepiece as spm
import jieba

langs_ID = {'zh': 128301, 'ko': 128302, 'vi': 128303,
            'de': 128317, 'en': 128318, 'nl': 128132,
            'ms': 128109, 'id': 128110, 'tl': 128111,
            'mn': 128103, 'my': 128104, 'th': 128105, 'lo': 128106, 'km': 128107,
            'lt': 128112, 'et': 128113, 'lv': 128133, 'hu': 128115,
            'pl': 128116, 'cs': 128117, 'sk': 128118, 'sl': 128119, 'hr': 128120, 'bs': 128121, 'sr': 128306, 'bg': 128304,
            'mk': 128122, 'ru': 128305, 'uk': 128307, 'be': 128123,
            'sq': 128124, 'el': 128125, 'ka': 128126, 'hy': 128127,
            'ro': 128108, 'fr': 128100, 'es': 128102, 'pt': 128101,
            'fa': 128310, 'he': 128311, 'ar': 128308, 'ps': 128309,
            'tr': 128128, 'kk': 128129, 'uz': 128130, 'az': 128131,
            'hi': 128315, 'ta': 128316, 'ur': 128313, 'bn': 128312, 'si': 128314, 'ne': 128114}

translate_ID = 128300


class SpmTokenizer(object):

    def __init__(self, model_file):

        self.sp = spm.SentencePieceProcessor(model_file=model_file)

        self.specialIDNum = 300
        self.eod_id = self.vocab_size - 1
        self.eot_id = self.vocab_size - 2
        self.pad_id = self.vocab_size - 3

        # langsList=['ar','bg','bs','cs','de','el','en','es','et','fa',
        #            'fr','he','hr','hu','id','it','nl','pl','pt','ru',
        #            'sl','tr','ur']
        # self.connectTxtToId = {}
        # i = 1
        # for lang in langsList:
        #     self.connectTxtToId[f'zh-{lang}'] = i
        #     i += 1
        #     self.connectTxtToId[f'{lang}-zh'] = i
        #     i += 1

    @property
    def vocab_size(self):
        return self.sp.vocab_size() + self.specialIDNum

    @property
    def spmVocabSize(self):
        return self.sp.vocab_size()

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text):
        """ Tokenize a string. """
        return self.sp.encode(text)

        # # Adapted parallel corpus
        # texts = text.split(' _??????_ ')
        # if len(texts) == 1:
        #     return self.sp.encode(text)
        # if len(texts) == 3:
        #     ids1 = self.sp.encode(texts[0])
        #     connectId = self.sp.vocab_size() + self.connectTxtToId[texts[1]]
        #     ids2 = self.sp.encode(texts[2])
        #     return ids1 + [connectId] + ids2
        # return []


    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        ids = [id if id < self.sp.vocab_size() else 0 for id in ids]
        return self.decode(ids)

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace('\u2583', '\n')
        return text


if __name__ == '__main__':
    Hindi_text = '?????????????????? ?????????????????? ???????????? ???????????????????????????????????? ???????????? ???????????????????????????????????? ???????????? ???????????? ???????????????????????????????????????????????? ??????????????? ?????????????????? ?????? ?????????????????????????????????????????? ?????????????????????????????????????????? ??????????????? ?????? ???????????????????????????????????????????????? ?????????????????????????????????????????????????????? ?????????????????? ???????????????????????????????????????????????? ??????????????? ?????????????????????????????? ?????????????????? ???????????? ?????????????????????????????? ?????????????????? ???????????? ?????????????????????????????????????????? ???????????????????????????????????????????????? ???????????????????????????????????????????????? ????????? ?????? ?????????????????????????????? ?????? ?????????????????????????????? ?????? ?????????????????????????????? ?????????????????? ???????????? ?????????????????????????????? ?????????????????? ???????????? ?????????????????????????????? ?????????????????? ???????????? ????????????????????????????????????????????? ????????????????????? ???????????????????????????????????? ?????? ???????????????'
    Chinese_text = '?????? ???\n???????????????-??????????????????-?????????????????????????????????????????????????????????????????????????????????????????????????????????-??????PK????????????????????????PK???????????????????????????????????????????????????????????????????????????2????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'
    Urdu_text = '????????????????????????? ???????? ?????????????? ???????????? ???? ?????????????? ???????????? ???????????? ???????????? ???????????????? ???????? ?????? ?????????? ?????????????? ???????? ???????? ???????? ???????? (??) ???? ???????????? ?????????? ?????????????????? ?????????????? ???????? ???????? ??????????????????? ???? ?????????? ???????? ???????? ???????? ?????? ???????? ?????????? ???? ?????????????? ???????????? ???????? ?????? ???? ???? ???????? ??????????? ???????? ???????? ?????????? ???????????? ?? ???????????? ???? ????????????????? ?????????? ???????????? ?????????????? ????????????????????????????????????????????????????????? ?????????????? ?? ???????????? ?? ???????? ??????????????? ???? ???????????? ???????????? ???????? ?? ?????? ?????????????? ???? ???? ?????????? ??????????.'
    Thai_text = '?????? VDO ??????????????????????????????????????????????????????????????????????????? Amibroker Backtest ?????? Backtest ????????????????????? Backtest ????????????????????????????????????????????????????????????????????????????????? ????????????????????????-????????? ?????????????????????????????????-????????????????????????????????????????????????????????????????????????????????????????????? ????????????????????????????????????????????????????????? ???????????????????????????????????????????????????????????????????????????????????????????????? ??????????????????????????????????????????????????????????????????????????????????????????????????????????????? ??????????????????????????????????????????????????? Amibroker ????????????????????????????????????????????????????????????????????????????????????????????? Backtest ??????????????? ??????????????????????????????????????????????????????????????????????????? ?????????????????????????????????????????????????????????????????????????????? windows ?????????????????? ????????????????????????????????????????????????????????????????????????????????????????????????????????? Backtest ??????????????????????????????????????????????????? 2 ????????????????????????????????????????????????????????? Analysis document ???????????????????????????????????????????????????????????????????????????????????? test ?????????????????????????????????????????????????????? ?????? test ???????????????????????????????????? ????????????????????????????????????????????????????????????????????????????????????????????? Analysis ??????????????????????????????????????? Backtest ?????????????????????????????????????????????????????? test ????????????????????? ???????????????????????????????????? ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? ????????????????????? test ???????????????????????????????????????????????????????????? ???????????????????????? test ???????????????????????????????????????????????????????????? test ????????????????????????????????????????????????????????????????????????????????????????????? set ?????????????????? test ??????????????????????????????????????? test bar ????????????????????????????????????????????? ??????????????????????????? test ??????????????????????????? from ????????? to ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? ????????????????????????????????????????????????????????????????????? ????????????????????????????????????????????????????????????????????? format ?????????????????????????????????????????????????????????????????????????????? ???????????????/?????????/???????????????????????????????????? format ????????????????????? ?????????????????????????????? ?????????/???????????????/?????? ???????????????????????????????????????????????? windows ???????????????????????????????????????????????????????????????????????????????????????????????? ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????2012 ???????????????????????????????????? ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? format ???????????????????????????????????????????????????????????? 2011 ????????????????????????????????????????????????????????? Analysis ????????????????????????????????????????????????????????????????????? windows ???????????? ????????????????????????????????? formula ????????????????????????????????????????????????????????????????????????????????????????????????????????? ?????????????????????????????????????????? ?????????????????????????????????????????????????????? new formula ????????????????????????????????? windows ?????????????????????????????? ???????????????????????????????????????????????????????????? ????????????????????????????????????????????? windows ????????????????????????????????????????????????????????????????????? windows ?????????????????????????????????????????? formula ?????????????????? formula ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? default ????????? Amibroker ?????????????????????????????? default ?????????????????? ????????????????????????????????????????????????????????? code formula ??????????????????????????????????????????????????????????????? ???????????????????????????????????????????????????????????? customs ??????????????????????????? ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????? ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? backtest ????????????????????????????????????????????? 2 ?????????????????????????????????????????? new Analysis ??? Analysis document ???????????? Formula 2 ???????????????????????????????????????????????????????????????????????????????????????????????? 2 ?????????????????????????????? ??????????????????????????????????????? ?????????????????????????????????????????? backtest ??????????????????????????????????????????????????????????????????????????????????????????????????????????????? level ????????? introduction ???????????? level basic advance ???????????? inter media ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? Ok ??????????????????'
    Malay_text = '????????? ????????????????????? ??????????????? ????????????????????? \n ?????????????????? ???????????????????????????????????????????????? ??????????????? ??????????????????????????????????????? ?????????????????????????????????. ????????????????????? ???????????? ?????????????????? ????????????????????????????????? ????????????????????????????????? ??????????????????????????????????????????????????????. ??????????????? ????????????????????????????????? ??????????????????????????????????????? ?????????????????? ????????????????????????????????? ?????????????????? ???????????????????????? ??????????????? ??????????????????????????? ???????????? ????????????????????????????????? ????????????????????? ????????????????????? ?????????????????????????????? ???????????? ?????????????????????. ????????????????????????????????????????????????????????? ?????????????????????????????? ????????????????????? ?????????????????? ?????????????????? ??????????????????????????? ???????????? ??????????????????????????????????????????????????????????????????????????? ???????????????????????? ?????????????????? ?????????????????? ????????????????????????????????? ?????????????????? ?????????????????? ???????????????. ????????????????????????????????????????????? ?????????????????????????????? ?????????????????????????????? ???????????????????????????????????????????????? ???????????????????????????????????? ????????????????????????????????? ??????????????????????????????. ?????????????????? ????????????????????????????????? ??????????????? ??????????????? ?????????????????????????????????.'
    Arabic_text = '???? ???????? ???? ???????? ???????? ?????? ?????????? ???????????????? ?????????? ???? ?????????? ???? ??????????????????????? ???????????????? ??????????????? ?? ?????????? ??????????????????? ?????????? ???????? ??????. ?????????? ?????? ?????????????? ?????????? ???? ????????????? ???????????????? ???????????? ???????????????? ????????????????????? ?? ???????????? ?????? ???????? ??????. ?????? ???? ???????? ?????? ???? ???????????? ?????????? ???? ???????????? ??????????????? ???????????? ???????????? ????????????????? ?????????? ???? ???????? ???? ???????? ????????????????? ?????????? ???????????? ??????????. ???? ?????? ?????????? ?????? ?????????? ???? ??????????? ????????????????? ???????????? ?????????? ?????????? ?????????? ?? ??????????????? ????????-??????????????? ?? ????????????????????? ?????? ???? ???????? ?? ????????????????? ??????????? ???????????????? ?????? ??????.'

    tokenizer = SpmTokenizer('spm.128k.model.1')
    tokens = tokenizer.tokenize(Chinese_text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    txt = tokenizer.convert_ids_to_tokens(ids)

    line1 = '34'
    line2 = '4434'
    a = f"{line1} _??????_ {'zh'}-{'ar'} _??????_ {line2}"
    b = tokenizer.tokenize(a)
    aa  = '??? ?????? ??? ?????? ?????? ???????????? ??? ?????? ??? ?????? ??? ?????? ?????? ??? ?????? ??? ?????? ?????? ?????? ??? ??? ?????? ??? ???????????? ?????? ?????? ??? ?????? ?????? ???????????? ??? ?????? ??? ?????? ?????? ?????? ??? ?????? ?????? ??? ?????? ???'
    tokens2 = tokenizer.tokenize(aa)
    tokens2 = [i for i in tokens2 if i != 119132]
    tokens3 = tokenizer.tokenize(''.join(aa.split()))
    tokens3 = [i for i in tokens3 if i != 119132]
    for i in tokens2:
        if i != 119132:
            print(tokenizer.convert_ids_to_tokens([i]))
    for i in tokens3:
        print(tokenizer.convert_ids_to_tokens([i]))
    aaa = ' '.join(jieba.cut(''.join(aa.split()).strip()))
    print(txt)

    pass





