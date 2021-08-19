import re
import emoji
import unidecode
from tqdm import tqdm

my_stopword_list = {'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'd', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'more', 'most', 'my', 'myself', 'no', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'she', "she's", 'should', "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'}
my_emoji_stopword = {'🏻','🏼','🏽','🏾','🏿','♂','♀'}
class Preprocess():
    def __init__(self):
        self.REGEX_REMOVE_PUNCTUATION = re.compile('[%s]' % re.escape('!¡"$%&\'()*+,.ªº/:;<=>#¿?[\\]^_`{|}~'))
        self.URL_RE = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
        self.WWW_RE = re.compile(r'www?[-_.?&~;+=/#0-9A-Za-z]{1,2076}')
        #self.HASHTAG_RE = re.compile(r'#[-_.?&~;+=/#0-9A-Za-z]{1,2076}')
        self.MENTION_RE = re.compile(r'@[-_.?&~;+=/#0-9A-Za-z]{1,2076}')
        self.MAIL_RE = re.compile(r'\S*@\S*\s?')
        #self.DIGIT_RE = re.compile(r"\b[0-9]+\b\s*")
        self.DIGIT_RE = re.compile(r"\S*\d\S*") #remove if contains digit
        self.REPEATED_LETTER =  re.compile(r"([a-q?=t-z])\1{1,}")
        self.REPEATED_LETTER_RS =  re.compile(r"([rs])\1{2,}")
        #self.REPEATED_LETTER_HIFEN = re.compile(r"([-?=t-z])\1{1,}") #remove repeated
        self.LETTER_HIFEN = re.compile(r"(?<!\w)\W+|\W+(?!\w)") #remove hifen if hifen is between and len(1) guarda-roupa (keep) meu--deus (remove)
        

            

    def extract_emojis(self, s):
        return [c for c in s if c in emoji.UNICODE_EMOJI]


    def remove_nprop(sentence):
        tags = tagger.tag(sentence.split())
        
        words = [t[0] for t in tags if t[1]!='NPROP']
        names = [t[0] for t in tags if t[1] =='NPROP']

        if(len(names)!=0):
            founded_names.append(names)

        return " ".join(words)
    def transform(self, texts):
        #texts: lista de posts
        #labels: lista com o sentimento de cada post
        new_texts = []
        for t in tqdm(range(len(texts)), total=len(texts), position=0):
            if "div>" not in texts[t] and "<img src=" not in texts[t] and "style=" not in texts[t] and "text-align:" not in texts[t]:
            
                sentence = texts[t].lower().replace("\n", "").replace("\t", " ").strip() 
                sentence = " ".join([s for s in sentence.split() if s not in ["rt", "RT"]]) 
                #sentence = self.remove_nprop(sentence)
                emojis = self.extract_emojis(sentence) #encontra os emojis
                emojis = [e for e in emojis if e not in my_emoji_stopword]
                sentence = unidecode.unidecode(sentence)
                sentence =  self.MAIL_RE.sub(" ", sentence)
                sentence =  self.URL_RE.sub(" ", sentence)
                sentence =  self.WWW_RE.sub(" ", sentence)
                #sentence =  self.HASHTAG_RE.sub(" ", sentence)
                sentence =  self.REPEATED_LETTER.sub(r"\1", sentence)
                sentence =  self.REPEATED_LETTER_RS.sub(r"\1\1", sentence)
                sentence =  self.MENTION_RE.sub(" ", sentence)
                sentence = self.LETTER_HIFEN.sub(" ", sentence)

                sentence =  self.REGEX_REMOVE_PUNCTUATION.sub(" ", sentence)
                sentence =  self.DIGIT_RE.sub("", sentence)


                sentence = " ".join(filter(lambda x:x.strip() not in my_stopword_list, sentence.split()))
                sentence = " ".join([sentence] + emojis) #adiciona os emojis novamente 

                    
                sentence = sentence.strip()
                new_texts.append(sentence)

        return new_texts
    
    def fit_transform(self, texts, labels, isTrain=False):
        #texts: lista de posts
        #labels: lista com o sentimento de cada post
        posts_dict = {}
        new_texts, new_labels = [], []
        print("preprocessing") 
        for t in tqdm(range(len(texts)), total=len(texts), position=0):
            if "div>" not in texts[t] and "<img src=" not in texts[t] and "style=" not in texts[t] and "text-align:" not in texts[t]:              
                sentence = texts[t].lower().replace("\n", "").replace("\t", " ").strip() 
                sentence = " ".join([s for s in sentence.split() if s not in ["rt", "RT"]])  
                #sentence = self.remove_nprop(sentence)
                emojis = self.extract_emojis(sentence) #find emojis 
                emojis = [e for e in emojis if e not in my_emoji_stopword]  #remove stopwords emojis
                sentence = unidecode.unidecode(sentence) #normalize
                sentence =  self.MAIL_RE.sub(" ", sentence) #remove email
                sentence =  self.URL_RE.sub(" ", sentence) #remove url
                sentence =  self.WWW_RE.sub(" ", sentence) #remove www
                #sentence =  self.HASHTAG_RE.sub(" ", sentence) #remove hashtags
                sentence =  self.REPEATED_LETTER.sub(r"\1", sentence)  
                sentence =  self.REPEATED_LETTER_RS.sub(r"\1\1", sentence)
                sentence =  self.MENTION_RE.sub(" ", sentence)
                sentence = self.LETTER_HIFEN.sub(" ", sentence)
                sentence =  self.REGEX_REMOVE_PUNCTUATION.sub(" ", sentence)
                sentence =  self.DIGIT_RE.sub("", sentence)


                sentence = " ".join(filter(lambda x:x.strip() not in my_stopword_list, sentence.split()))
                sentence = " ".join([sentence] + emojis) #adiciona os emojis novamente 

                    
                sentence = sentence.strip()

                if(sentence != ""):
                    posts_dict[sentence]=labels[t]
                    new_texts.append(sentence)
                    new_labels.append(labels[t])

        if(isTrain):
            return new_texts, new_labels
        else:
            return [d[0] for d in list(posts_dict.items())], [d[1] for d in list(posts_dict.items())]

 
if __name__ == '__main__':
    import pickle, lzma, random
    p = Preprocess()
    print(p.transform(['😋😋😋😋😋 meu amigaaao 👎🏾  🤦🏽‍♀️ 🤦🏿 🤷🏾‍♂️ aaaeee http://google.com google.com www.google.com']))