import os
import statistics

from tqdm import tqdm
import pickle

from sentence_transformers import SentenceTransformer, util


class AugmentorTester():
    def __init__(self, user2keyword, scale_factor=5, clusteridx='0', version=None):

        model_name_or_path = "multi-qa-MiniLM-L6-cos-v1"

        if version is not None:
            self.user_embedding_path = "./cache/user_embed-{}" + f"-{clusteridx}" + f"-v{version}" + ".pickle"
        else:
            self.user_embedding_path = "./cache/user_embed-{}" + f"-{clusteridx}" + ".pickle"

        self.scale_factor = scale_factor

        self.user2keyword = user2keyword
        self.num_users = len(self.user2keyword)

        self.model = SentenceTransformer(model_name_or_path)

        # save user embeddings
        self.user2embeddings = dict()

        if os.path.exists(self.user_embedding_path.format(str(self.num_users))):
            with open(self.user_embedding_path.format(str(self.num_users)), "rb") as fr:
                self.user2embeddings = pickle.load(fr)

        else:
            for user_id in tqdm(self.user2keyword, desc='creating user embeddings'):
                user_embedding = self.make_userEmbedding(user_id)
                self.user2embeddings[user_id] = user_embedding
            with open(self.user_embedding_path.format(str(self.num_users)), "wb") as fw:
                pickle.dump(self.user2embeddings, fw)

    def evaluate(self, original_text, augmented_text):
        origin_embed = self.make_docEmbedding(original_text)
        augmented_embed = self.make_docEmbedding(augmented_text)

        origin_creds = []
        augmented_creds = []

        for user_id in self.user2keyword:
            user_embedding = self.user2embeddings[user_id]

            origin_cred = self.calculate_credibility(origin_embed, user_embedding)
            augmented_cred = self.calculate_credibility(augmented_embed, user_embedding)

            origin_creds.append(origin_cred)
            augmented_creds.append(augmented_cred)

        origin_mean, origin_std = self.calculate_meanStd(origin_creds)
        augmented_mean, augmented_std = self.calculate_meanStd(augmented_creds)

        return (origin_mean, origin_std, augmented_mean, augmented_std)

    def make_userEmbedding(self, user_id, max_length=512):
        keywords = ' '.join(self.user2keyword[user_id])
        user_embedding = self.model.encode(keywords)
        return user_embedding

    def calculate_credibility(self, docEmbedding, userEmbedding):
        return util.dot_score(docEmbedding, userEmbedding).tolist()[0][0] * self.scale_factor

    def calculate_meanStd(self, credibilities):
        return round(statistics.mean(credibilities), 3), round(statistics.stdev(credibilities), 3)

    def make_docEmbedding(self, text, max_length=512):
        text_embedding = self.model.encode(text)
        return text_embedding


if __name__ == '__main__':
    user2keyword = {
        'user_id1': ["word1", "word2", "word3"],
        'user_id2': ["word4", "word5", "word3"],
        'user_id3': ["word1", "word6", "word3"]
    }

    tester = AugmentorTester(
        user2keyword=user2keyword,
    )

    # results will be (origin_mean, origin_std, augmented_mean, augmented_std)
    results = tester.evaluate("word1 is student", "word1 is student and word2 with word 3")
